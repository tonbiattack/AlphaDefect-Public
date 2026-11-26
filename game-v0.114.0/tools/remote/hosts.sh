#!/usr/bin/env bash
# 租用机登记（tools/remote/hosts.json）与训练线登记（~/.local/share/sls2/runs/remote/lines.json）的读取函数，
# 以及迁移、分叉、重启共用的几段操作。供本目录的本机脚本 source，不单独运行：
#   . "$(dirname "$0")/hosts.sh"
#   host_get a port                 # 字段值：PORT_A
#   host_ssh a                      # ssh -p PORT_A -i ~/.ssh/... -o IdentitiesOnly=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 root@rented-host.example
#   $(host_ssh a) 'ls'              # 在 A 上执行命令
#   rsync -a -e "$(host_rsync_e a)" "$(host_addr a):$(host_get a run_dir)/eval.jsonl" .
#   host_pending b && echo 待填      # pending 为 true 或 ssh_host 未填时成立
#   main_host; challenger_host      # 主线、对照线所在的主机（lines.json 不存在时主线为 a、没有对照线）
# 跳转：ssh 按主机名（ssh_host）匹配 ~/.ssh/config，A 的伦敦跳转与连接复用写在那里；B 若也要跳转，须在 ~/.ssh/config 给它的主机名加同样的条目。
# 测试时可用环境变量 HOSTS_JSON、LINES_JSON 指向别的登记文件。

REMOTE_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$REMOTE_TOOLS/../.." && pwd)
HOSTS_JSON=${HOSTS_JSON:-$REMOTE_TOOLS/hosts.json}
REMOTE_LOCAL=${REMOTE_LOCAL:-$HOME/.local/share/sls2/runs/remote}
LINES_JSON=${LINES_JSON:-$REMOTE_LOCAL/lines.json}
# 游戏版本只写在 engine/game_version.txt（引擎模块与游戏更新适配.md）；文件不在时退回 v0.111.0
GAME_VER=${GAME_VER:-$(tr -d '[:space:]' < "$REPO/engine/game_version.txt" 2>/dev/null || echo v0.111.0)}
# 与机器配置相关的训练参数：迁移、分叉时默认沿用目标机原来的值（目标机没有旧配置时沿用来源的值），不算作“改动”
HW_KEYS='["actors", "envs_per_actor", "actor_device", "learner_device", "actor_threads", "learner_threads", "infer_wait_ms", "eval_workers", "eval_cpu_cores", "eval_envs", "eval_device"]'

die() { echo "错误：$*" >&2; exit 1; }

_hosts_py() {
  python3 - "$HOSTS_JSON" "$@" <<'EOF'
import json, sys
path, act, *a = sys.argv[1:]
hosts = json.load(open(path, encoding="utf-8"))["hosts"]
pend = lambda h: bool(h.get("pending")) or h.get("ssh_host") in (None, "", "待填")
if act == "names":
    print("\n".join(hosts)); sys.exit()
if act == "active":
    print("\n".join(n for n, h in hosts.items() if not pend(h))); sys.exit()
if a[0] not in hosts:
    print(f"hosts.json 里没有主机 {a[0]}", file=sys.stderr); sys.exit(2)
h = hosts[a[0]]
if act == "pending":
    sys.exit(0 if pend(h) else 1)
v = h.get(a[1], "")
print(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, bool)) else v)
EOF
}

host_names()  { _hosts_py names; }                       # 全部主机名，每行一个
host_active() { _hosts_py active; }                      # 不是待填的主机名
host_exists() { _hosts_py names | grep -qx -- "$1"; }
host_get()    { _hosts_py get "$1" "$2"; }               # host_get 名 字段
host_pending() { _hosts_py pending "$1"; }               # 返回 0 表示待填（或不存在时返回 2）
host_label()  { host_get "$1" label; }
host_key()    { local k; k=$(host_get "$1" key); echo "${k/#\~/$HOME}"; }
host_addr()   { echo "$(host_get "$1" user)@$(host_get "$1" ssh_host)"; }
host_run()    { host_get "$1" run_dir; }                 # 远端运行目录（相对远端家目录）
host_deploy() { host_get "$1" deploy; }                  # 远端仓库目录（绝对路径）
host_local()  { echo "$REMOTE_LOCAL/$(host_get "$1" local)"; }    # 本机拉取目录
# 拉取状态文件的后缀：A 沿用原来的 pull.ok / pull.err（看板现在读它们），其他主机为 pull-<名>.ok / pull-<名>.err
host_pull_tag() { [ "$1" = a ] || echo "-$1"; }
# 给 rsync -e 用的 ssh 命令（不含 用户@主机）
host_rsync_e() {
  local k; k=$(host_key "$1")
  echo "ssh -p $(host_get "$1" port)${k:+ -i $k -o IdentitiesOnly=yes} -o ConnectTimeout=30 -o ServerAliveInterval=30"
}
host_ssh() { echo "$(host_rsync_e "$1") $(host_addr "$1")"; }    # 完整 ssh 命令前缀
# 主机必须存在且已登记地址，否则报错退出（脚本绝不尝试连接待填的主机）
host_require() {
  host_exists "$1" || die "hosts.json 里没有主机 $1（有：$(host_names | tr '\n' ' ')）"
  if host_pending "$1"; then die "主机 $1（$(host_label "$1")）还没登记地址（hosts.json 里 pending），不连接"; fi
}

# ---------- 训练线登记 lines.json ----------
_lines_py() {
  python3 - "$LINES_JSON" "$@" <<'EOF'
import json, os, sys, time
path, act, *a = sys.argv[1:]
try:
    d = json.load(open(path, encoding="utf-8"))
except FileNotFoundError:
    d = {}
d.setdefault("main", {"host": "a"}); d.setdefault("challenger", None)
if act == "main":
    print(d["main"].get("host") or "a")
elif act == "challenger":
    print((d["challenger"] or {}).get("host", ""))
elif act == "show":
    print(json.dumps(d, ensure_ascii=False, indent=1))
elif act in ("set_main", "set_challenger"):
    key = act[4:]
    if d.get(key):     # 被替换的记录留在 history 里，不丢
        d.setdefault("history", []).append({"role": key, "ended": time.time(), "end_reason": a[1] if len(a) > 1 else "", **d[key]})
    d[key] = json.loads(a[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)
EOF
}
main_host()       { _lines_py main; }
challenger_host() { _lines_py challenger; }
lines_show()      { _lines_py show; }
lines_set_main()       { _lines_py set_main "$1" "${2:-}"; }        # JSON 记录 [被替换的原因]
lines_set_challenger() { _lines_py set_challenger "$1" "${2:-}"; }  # JSON 记录或 null [原因]

# ---------- 远端操作片段（在远端运行目录里执行） ----------
# alive：训练学习进程在跑（pid 文件里的进程存在且确实是 train_defect.py，防止 pid 被别的进程复用）
REMOTE_ALIVE='alive() { [ -f learner.pid ] && P=$(cat learner.pid) && kill -0 "$P" 2>/dev/null && grep -qa train_defect "/proc/$P/cmdline"; }'
# 停训练：先放 STOP（守护不再拉起），再 TERM 学习进程（会先存检查点），最多等 3 分钟
REMOTE_STOP='touch STOP; if alive; then kill -TERM "$P"; for i in $(seq 90); do kill -0 "$P" 2>/dev/null || break; sleep 2; done; kill -0 "$P" 2>/dev/null && { echo "学习进程停不下来（STOP 仍在）"; exit 1; }; fi; echo 训练已停'

# 同步本机代码到主机：python/、远端运行的守护与备份脚本；ENGINE=1 时另同步引擎源码与 catalog（需随后在远端编译，见 remote_build）
sync_code() {
  local h=$1 E A D; E=$(host_rsync_e "$h"); A=$(host_addr "$h"); D=$(host_deploy "$h")
  rsync -a --exclude __pycache__ -e "$E" "$REPO/python/" "$A:$D/python/"
  rsync -a -e "$E" "$REMOTE_TOOLS/ckpt_backup.sh" "$REMOTE_TOOLS/train_watchdog.sh" "$REMOTE_TOOLS/start_train.sh" \
    "$REMOTE_TOOLS/stoploss.sh" "$A:$D/tools/remote/"
  if [ -n "${ENGINE:-}" ]; then
    rsync -a --exclude bin/ --exclude obj/ -e "$E" "$REPO/engine/Sts2Headless/" "$A:$D/engine/Sts2Headless/"
    rsync -a -e "$E" "$REPO/engine/catalog/" "$A:$D/engine/catalog/"
    # 游戏版本（Python 按它找内容目录 engine/catalog/<版本>/；不在时退回 v0.111.0，2026-11-13 Fool 首次部署因此用错内容目录）
    [ -f "$REPO/engine/game_version.txt" ] && rsync -a -e "$E" "$REPO/engine/game_version.txt" "$A:$D/engine/"
    # Rust 引擎（配置 engine=rust 时采集用）：本机编好的 musl 静态版（目标机的 glibc 可能较旧），
    # 先传成 .new，训练停下之后由 remote_build 换上（运行中的采集进程不会换到新程序）
    local RB=$REPO/rust-engine/target/x86_64-unknown-linux-musl/release/sts2-serve
    if [ -x "$RB" ]; then
      $(host_ssh "$h") "mkdir -p $D/rust-engine/target/release"
      rsync -a -e "$E" "$RB" "$A:$D/rust-engine/target/release/sts2-serve.new"
    else
      echo "提示：没有 musl 版 Rust 引擎（cd rust-engine && cargo build --release -p sts2-serve --target x86_64-unknown-linux-musl），这次不更新" >&2
    fi
  fi
}
# 远端编译引擎的命令片段（ENGINE=1 时非空；训练停下之后才能执行，运行中的引擎还在用旧程序集）
remote_build() {
  [ -n "${ENGINE:-}" ] || { echo ':'; return; }
  local D; D=$(host_deploy "$1")
  echo "(export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_CLI_UI_LANGUAGE=en; cd $D/engine/Sts2Headless && dotnet build -c Release -p:Sts2GameDir=\$HOME/.local/share/sls2/game/$GAME_VER > ${D%/*}/build.log 2>&1) || { tail -20 ${D%/*}/build.log; echo 引擎编译失败，训练未启动（STOP 仍在）; exit 1; }; echo 引擎已编译; \
if [ -f $D/rust-engine/target/release/sts2-serve.new ]; then mv $D/rust-engine/target/release/sts2-serve.new $D/rust-engine/target/release/sts2-serve && echo Rust 引擎已更新; fi"
}
# 只读比较：本机代码与主机上的代码有哪些文件不同（rsync 预演，不写远端）。输出不同的文件，没有输出表示一致
code_diff() {
  local h=$1 E A D; E=$(host_rsync_e "$h"); A=$(host_addr "$h"); D=$(host_deploy "$h")
  { rsync -nrci --exclude __pycache__ -e "$E" "$REPO/python/" "$A:$D/python/" | sed 's#^#python/ #'
    rsync -nrci --exclude bin/ --exclude obj/ -e "$E" "$REPO/engine/Sts2Headless/" "$A:$D/engine/Sts2Headless/" | sed 's#^#engine/ #'
    rsync -nrci -e "$E" "$REPO/engine/catalog/" "$A:$D/engine/catalog/" | sed 's#^#catalog/ #'
  } | grep -v ' \.d' || true
}
# 只读检查目标机能否跑训练：部署目录、虚拟环境、引擎程序集、守护定时任务、磁盘余量、训练是否在跑。
# 输出若干行说明；最后一行为 ALIVE=0/1
preflight() {
  local h=$1 R D; R=$(host_run "$h"); D=$(host_deploy "$h")
  $(host_ssh "$h") "$REMOTE_ALIVE
    for f in $D/.venv/bin/python $D/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll; do [ -e \$f ] || echo \"缺少 \$f\"; done
    crontab -l 2>/dev/null | grep -q train_watchdog.sh || echo '提示：crontab 里没有 train_watchdog.sh（训练进程退出后不会自动拉起）'
    crontab -l 2>/dev/null | grep -q ckpt_backup.sh || echo '提示：crontab 里没有 ckpt_backup.sh（没有定时备份）'
    echo \"磁盘余量 \$(df -BG --output=avail \$HOME | tail -1 | tr -dc 0-9) GB\"
    cd $R 2>/dev/null && alive && echo ALIVE=1 || echo ALIVE=0"
}

# 从主机取检查点到本机目录 STAGE：STAGE/latest.pt、STAGE/train_config.json。设置 CKPT_VERSION、CKPT_TIME（检查点写出的时刻）、CKPT_SRC。
#   fetch_ckpt 主机 来源 STAGE TAG [dry]
# 来源：backup（默认）取最近一次 ckpt_backup 的备份；fresh 先在该主机上执行一次带 TAG 的备份（不停训练）再取；
#       latest 直接取运行目录的 latest.pt（训练进程先写 latest.tmp 再改名，正在训练时取也不会读到半个文件，但配置可能比检查点新）。
# dry 非空时只做只读查询并打印计划。
fetch_ckpt() {
  local h=$1 src=$2 stage=$3 tag=$4 dry=${5:-} R D SSH E A f cfg info
  R=$(host_run "$h"); D=$(host_deploy "$h"); SSH=$(host_ssh "$h"); E=$(host_rsync_e "$h"); A=$(host_addr "$h")
  case $src in
    fresh)
      if [ -n "$dry" ]; then
        echo "  [预演] 将在 $h 上执行一次备份（不停训练）：TAG=$tag RUN_DIR=\$HOME/$R bash $D/tools/remote/ckpt_backup.sh；下面列出的是目前最近的一份"
      else
        $SSH "TAG=$tag RUN_DIR=\$HOME/$R bash $D/tools/remote/ckpt_backup.sh" || die "在 $h 上备份失败"
      fi ;;
    backup|latest) ;;
    *) die "检查点来源只能是 backup、fresh、latest：$src" ;;
  esac
  if [ "$src" = latest ]; then
    info=$($SSH "f=$R/latest.pt; [ -f \$f ] && echo \$f && stat -c %Y \$f") || die "读取 $h 失败"
  else
    info=$($SSH "f=\$(ls -1t $R/backups/v*.pt 2>/dev/null | head -1); [ -n \"\$f\" ] && echo \$f && stat -c %Y \$f") || die "$h 上没有备份（$R/backups/v*.pt）"
  fi
  f=$(echo "$info" | sed -n 1p); CKPT_TIME=$(echo "$info" | sed -n 2p); CKPT_SRC="$h:$f"
  [ -n "$f" ] || die "$h 上没有检查点"
  if [ "$src" = latest ]; then cfg=$R/train_config.json; CKPT_VERSION=; else cfg=${f%.pt}.train_config.json; CKPT_VERSION=$(basename "$f" | sed -n 's/^v\([0-9]*\)-.*/\1/p'); fi
  echo "  检查点：$CKPT_SRC（写出于 $(date -d @"$CKPT_TIME" '+%F %T %Z')${CKPT_VERSION:+，版本 $CKPT_VERSION}），配置：$cfg"
  [ -n "$dry" ] && return 0
  mkdir -p "$stage"
  rsync -a --partial -e "$E" "$A:$f" "$stage/latest.pt" || die "复制检查点失败"
  rsync -a -e "$E" "$A:$cfg" "$stage/train_config.json" 2>/dev/null || {
    echo "  备份没有配套的配置文件，改取当前的 train_config.json"; rsync -a -e "$E" "$A:$R/train_config.json" "$stage/train_config.json" || die "复制配置失败"; }
  # 本机读一遍检查点：确认文件完整，并取到版本号
  local PY=$REPO/.venv/bin/python v
  if [ -x "$PY" ]; then
    v=$("$PY" -I -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["version"])' "$stage/latest.pt" 2>/dev/null) \
      || die "本机读不出复制来的检查点 $stage/latest.pt"
    [ -n "$CKPT_VERSION" ] && [ "$CKPT_VERSION" != "$v" ] && echo "  注意：文件名版本 $CKPT_VERSION 与检查点内版本 $v 不同，以后者为准"
    CKPT_VERSION=$v
  fi
  echo "  已取到本机 $stage（版本 ${CKPT_VERSION:-未知}）"
}

# 修改本机暂存的 train_config.json：先按 HW（与机器相关的键）覆盖，再按 CHANGE 覆盖。打印 CHANGE 各键改动前的值（JSON）。
#   edit_config 文件 CHANGE_JSON HW_JSON
edit_config() {
  python3 - "$1" "$2" "$3" "$HW_KEYS" <<'EOF'
import json, sys
path, change, hw, keys = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), set(json.loads(sys.argv[4]))
c = json.load(open(path, encoding="utf-8"))
before = {k: c.get(k) for k in change}
c.update({k: v for k, v in hw.items() if k in keys}); c.update(change)
json.dump(c, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(before, ensure_ascii=False))
EOF
}
# 取 JSON 或 @文件（文件时只取其中与机器相关的键），输出 JSON 对象；校验失败退出
read_hw() {
  # 从文件（某台机器的整份配置）读时，文件里没写的机器参数补上默认值：否则会沿用来源机器的值
  # （2026-10-26 采纳对照线时，b 的旧配置没有 eval_device，结果带上了 a 的 cuda）
  local defaults='{}'
  if [[ $1 == @* ]] && [ -x "$REPO/.venv/bin/python" ]; then
    defaults=$("$REPO/.venv/bin/python" -I -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from sts2rl.config import TrainConfig
c = TrainConfig(); print(json.dumps({k: getattr(c, k) for k in json.loads(sys.argv[2]) if hasattr(c, k)}))' "$REPO/python" "$HW_KEYS") || defaults='{}'
  fi
  python3 - "$1" "$HW_KEYS" "$defaults" <<'EOF' || die "--hw 不是 JSON 对象或文件：$1"
import json, sys
s, keys, defaults = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
d = json.load(open(s[1:], encoding="utf-8")) if s.startswith("@") else json.loads(s or "{}")
assert isinstance(d, dict)
print(json.dumps({k: d[k] if k in d else defaults[k] for k in keys if k in d or k in defaults}, ensure_ascii=False))
EOF
}
# 校验 JSON 对象（改动参数），通过时原样输出
check_json_obj() {
  python3 -c 'import json, sys; d = json.loads(sys.argv[1]); assert isinstance(d, dict); print(json.dumps(d, ensure_ascii=False))' "$1" 2>/dev/null \
    || die "不是 JSON 对象：$1"
}
