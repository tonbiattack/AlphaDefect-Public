"""训练看板的只读价值诊断卡片；只读取本机汇总，不运行推理或训练。"""
import html
import json
from pathlib import Path
import time

KINDS={'2':'删牌／升级选牌','3':'抓牌奖励','4':'卡包','5':'领奖励','6':'事件','7':'休息处','8':'商店','9':'宝箱','10':'小游戏','11':'路线'}


def fmt(x,d=3):return '—' if x is None else f'{x:,.{d}f}'
def esc(x):return html.escape(str(x))


def table(headers,rows):
    return '<div class="tscroll"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{esc(x)}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def value_probe_section(root,panel,chart=None):
    root=Path(root)
    try:s=json.loads((root/'summary.json').read_text())
    except (OSError,ValueError):s={}
    if not s:
        body=panel('价值诊断采集','统计不会进入策略、奖励或优化器。','<div class="empty">等待第一批诊断数据同步到本机。</div>','span3')
        return '<details class="sec" data-sec="value-probe" data-open="desktop"><summary class="sechead"><h2>局外价值诊断</h2></summary><div class="row g3">'+body+'</div></details>'
    counts=s.get('all_counts',{});latest=s.get('latest_time');age=max(0,time.time()-latest) if latest else None
    status='采集中' if age is not None and age<600 else '等待新数据' if latest else '尚无数据'
    if s.get('activation_pending'):status='新探针等待启用；现有统计正在留存'
    statuses=s.get('statuses',[]);active=[x for x in statuses if time.time()-x.get('time',0)<3600]
    dropped=sum(x.get('dropped',0) for x in active);errors=sum(x.get('errors',0) for x in active)
    lines=[('状态',status),('最近诊断版本',s.get('latest_version') or '—'),('累计批次记录',counts.get('batch',0)),
           ('累计抽样完整局',counts.get('episode',0)),('本机原始文件',s.get('archive_files',0)),
           ('本机原始数据',f"{s.get('archive_bytes',0)/(1<<20):,.1f} MB"),('单次探针耗时',fmt(s.get('probe_seconds_mean'))+' 秒'),
           ('近一小时生产者：丢弃／写入错误',f'{dropped} / {errors}')]
    if latest:lines.append(('数据时间',time.strftime('%m-%d %H:%M:%S',time.localtime(latest))))
    legacy=s.get('legacy_latest') or {}
    if legacy:
        lines.append(('既有健康统计留存',counts.get('legacy',0)))
        lines.append(('旧口径局外 EV',fmt((legacy.get('health') or {}).get('ev',{}).get('macro'))))
    if s.get('sync'):lines.append(('最近同步','成功' if s['sync'].get('ok') else '部分来源等待重试'))
    body=panel('数据采集与本机留存','完整局独立抽样；统计和原始观测保存在本机 runs/value_probe。未完成或卡住的轨迹不冒充终局标签。',table(['项目','读数'],lines))
    old=s.get('old_groups',{});rows=[]
    for k,name in [('macro','全部局外'),('macro/multi/1','局外·多选'),('macro/multi/0','局外·单选'),('combat/mode/1','整局战斗'),('combat/mode/0','战斗课程')]:
        a=old.get(k,{})
        rows.append([name,fmt(a.get('n'),0),fmt(a.get('ev')),fmt(a.get('mse')),fmt(a.get('bias')),fmt(a.get('target_var'))])
    body+=panel('解释方差、误差与目标方差',f"最近 {s.get('window_batches',0)} 个诊断批次按统计量合并；偏差 = 预测 − 目标。EV 不反映常数偏差。",table(['范围','步数','EV','MSE','偏差','目标方差'],rows),'span2')
    rows=[]
    for k,name in [(f'macro/act/{a}',f'第 {a} 幕') for a in (1,2,3)]+[(f'macro/kind/{k}',v) for k,v in KINDS.items()]+[('macro/ancient/1','先古遗物选择')]:
        a=old.get(k)
        if a:rows.append([name,fmt(a['n'],0),fmt(a.get('ev')),fmt(a.get('mse')),fmt(a.get('bias')),fmt(a.get('residual_var'))])
    body+=panel('按幕与决策类型定位','各行可能重叠，不能相加；目标方差过小时 EV 显示为“—”，仍保留绝对误差。',table(['分组','步数','EV','MSE','偏差','残差方差'],rows),'span2')
    rows=[]
    for k,name in [('old','采集时旧预测'),('current','更新前当前预测'),('public','当前预测·去特权')]:
        a=s.get('paired',{}).get(k,{}).get('macro',{})
        rows.append([name,fmt(a.get('n'),0),fmt(a.get('ev')),fmt(a.get('mse')),fmt(a.get('bias'))])
    sign=s.get('advantage_sign',{});n=sign.get('macro_n',0)
    extra=f'<p class="psub">换用当前价值后，优势符号变化：{fmt(100*sign.get("old_current_changes",0)/n,2) if n else "—"}%。去特权前后策略分数最大差：{fmt(s.get("privileged_policy_max_delta"),6)}。</p>'
    body+=panel('同一批局面的价值对照','三行使用相同样本、相同自举目标，额外前向使用 FP32。去特权是输入消融，不是重新训练的公开价值头。GPU 重复前向也可能出现微小分数差，非零值须与重复计算对照。',table(['预测','步数','EV','MSE','偏差'],rows)+extra)
    targets=s.get('episode_targets',{});rows=[]
    for k,name in [('segment','原分段目标'),('full_lambda','跨完整局 λ 目标'),('full_lambda_no_cut','完整局 λ·不截探索'),('mc_behavior','实际完整回报')]:
        a=targets.get(k,{}).get('macro',{});rows.append([name,fmt(a.get('ev')),fmt(a.get('mse')),fmt(a.get('bias'))])
    h=s.get('horizon',{});ci=s.get('mc_cluster_ci',{}).get('ev')
    extra=f'<p class="psub">有效完整局 {s.get("valid_mc_episodes",0):,}；到段末平均 {fmt(h.get("mean_to_boundary"),1)} 个局外决策。完整回报 EV 的整局重采样区间：{esc("—" if not ci else f"{ci[0]:.3f}～{ci[1]:.3f}")}。</p>'
    body+=panel('跨段目标与完整回报','都比较采集时旧价值。完整回报来自训练行为序列，可能跨权重版本；不等同于固定策略的独立续局评估。',table(['目标','EV','MSE','偏差'],rows)+extra)
    g=s.get('gradients') or {};rows=[]
    names={'macro_value':'局外价值','combat_value':'战斗价值','policy':'策略项','other':'其他训练项'}
    for k,x in g.get('norm',{}).items():rows.append([names.get(k,k)+'梯度范数',fmt(x)])
    if 'macro_head_norm' in g:rows.append(['局外价值输出头梯度范数',fmt(g['macro_head_norm'])])
    for k,x in g.get('weighted_value_norm',{}).items():rows.append([('局外' if k=='macro' else '战斗')+'价值·按实际 loss 权重',fmt(x)])
    for k,x in g.get('cosine',{}).items():rows.append([' × '.join(names.get(z,z) for z in k.split('__'))+'方向余弦',fmt(x)])
    body+=panel('共享参数的梯度关系',f"第 {s.get('gradient_version') or '—'} 版的小样本只读探针。余弦为负表示方向相反；不写入训练梯度，也不执行优化器更新。",table(['项目','读数'],rows) if rows else '<div class="empty">等待梯度探针。</div>')
    cal=s.get('calibration',{}).get('mc_behavior') or {};rows=[]
    for b in cal.get('bins',[]):
        if b['n']:rows.append([fmt(b['n'],0),fmt(b['predicted']),fmt(b['target']),fmt(b['mse'])])
    base=s.get('baseline') or {};bst=base.get('test',{})
    text=f'<p class="psub">进度／资源线性基线：{esc("已按整局划分拟合" if base.get("ready") else "等待至少 100 个训练局、20 个验证局和 20 个测试局")}'
    if base.get('ready'):text+=f'；测试 MSE {fmt(bst.get("baseline",{}).get("mse"))}，采集价值 {fmt(bst.get("actor_value",{}).get("mse"))}'
    text+='。划分只用于离线诊断，不排除 PPO 样本。</p>'
    body+=panel('实际回报校准与简单基线','按旧价值的固定区间分箱。这里预测的是奖励总和，不是通关概率。',table(['步数','平均预测','平均回报','MSE'],rows)+text)
    ext=s.get('external') or {};ceiling=ext.get('noise_ceiling') or {};rank=ext.get('ranking') or {}
    rows=[['固定策略续局根数',ext.get('roots',0)],['独立续局次数',ext.get('rollouts',0)],
          ['噪声上限估计',fmt(ceiling.get('ev_max_estimate'))],['反事实候选比较根数',rank.get('roots',0)],
          ['用独立世界验证的根数',rank.get('validated_roots',0)]]
    body+=panel('独立续局与候选反事实','根状态导出、标签导入与统计接口已提供。需要额外固定策略续局；没有标签时不使用在线 EV 代填噪声上限。',table(['项目','读数'],rows),'span2')
    return '<details class="sec" data-sec="value-probe" data-open="desktop"><summary class="sechead"><h2>局外价值诊断</h2><span class="psub">只读采集 · 分层误差 · 跨段监督 · 梯度与独立评估</span></summary><div class="row g3">'+body+'</div></details>'
