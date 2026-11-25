# AlphaDefect ランチャー（Windows の GUI）
#
# ダブルクリックで起動する。WSL2 の中の ./play.sh を呼び出すだけで、
# ゲームの起動・推論サーバ・Mentor 連携はこれまでと同じ経路を使う。
#
#   launcher.cmd をダブルクリック（推奨）
#   または: powershell -NoProfile -ExecutionPolicy Bypass -STA -File launcher.ps1

Add-Type -AssemblyName System.Windows.Forms, System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

# このスクリプトの場所から、WSL 側のリポジトリのパスを作る
$here = $PSScriptRoot
if (-not $here) { $here = (Get-Location).Path }
$linuxRepo = "/home/teni2/AlphaDefect-Public"
if ($here -match '^\\\\wsl(?:\.localhost)?\\[^\\]+\\(.+)$') {
    $linuxRepo = "/" + ($matches[1] -replace '\\', '/')
} elseif ($here -match '^([A-Za-z]):\\(.+)$') {
    # Windows 側にコピーして使う場合は、既定の場所を使う（WSL のパスは決め打ち）
    $linuxRepo = "/home/teni2/AlphaDefect-Public"
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "AlphaDefect ランチャー"
$form.Size = New-Object System.Drawing.Size(760, 620)
$form.StartPosition = "CenterScreen"
$form.Font = New-Object System.Drawing.Font("Meiryo UI", 9)

function New-Label($text, $x, $y, $w = 90) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text = $text; $l.Location = New-Object System.Drawing.Point($x, $y)
    $l.Size = New-Object System.Drawing.Size($w, 22)
    return $l
}

# ---- モード
$gbMode = New-Object System.Windows.Forms.GroupBox
$gbMode.Text = "モード"; $gbMode.Location = New-Object System.Drawing.Point(14, 10)
$gbMode.Size = New-Object System.Drawing.Size(710, 70)
$rbNormal = New-Object System.Windows.Forms.RadioButton
$rbNormal.Text = "通常プレイ（MOD なし）"; $rbNormal.Location = New-Object System.Drawing.Point(14, 28)
$rbNormal.AutoSize = $true
$rbAI = New-Object System.Windows.Forms.RadioButton
$rbAI.Text = "AI 観戦（独自 MOD）"; $rbAI.Location = New-Object System.Drawing.Point(210, 28)
$rbAI.AutoSize = $true
$rbMentor = New-Object System.Windows.Forms.RadioButton
$rbMentor.Text = "Mentor 観戦（spire-mentor）"; $rbMentor.Location = New-Object System.Drawing.Point(390, 28)
$rbMentor.AutoSize = $true
$rbMentor.Checked = $true
$gbMode.Controls.AddRange(@($rbNormal, $rbAI, $rbMentor))
$form.Controls.Add($gbMode)

# ---- 設定
$gbSet = New-Object System.Windows.Forms.GroupBox
$gbSet.Text = "設定"; $gbSet.Location = New-Object System.Drawing.Point(14, 88)
$gbSet.Size = New-Object System.Drawing.Size(710, 150)

$cbChar = New-Object System.Windows.Forms.ComboBox
$cbChar.DropDownStyle = "DropDownList"
[void]$cbChar.Items.AddRange(@("auto", "DEFECT", "REGENT", "SILENT", "select"))
$cbChar.SelectedIndex = 0
$cbChar.Text = "auto"
$cbChar.Location = New-Object System.Drawing.Point(80, 28)
$cbChar.Size = New-Object System.Drawing.Size(150, 24)

$numAsc = New-Object System.Windows.Forms.NumericUpDown
$numAsc.Minimum = 0; $numAsc.Maximum = 10; $numAsc.Value = 0
$numAsc.Location = New-Object System.Drawing.Point(330, 28)
$numAsc.Size = New-Object System.Drawing.Size(60, 24)

$txtSeed = New-Object System.Windows.Forms.TextBox
$txtSeed.Location = New-Object System.Drawing.Point(480, 28)
$txtSeed.Size = New-Object System.Drawing.Size(120, 24)

$chkRandom = New-Object System.Windows.Forms.CheckBox
$chkRandom.Text = "乱数"; $chkRandom.Checked = $true
$chkRandom.Location = New-Object System.Drawing.Point(610, 28)
$chkRandom.AutoSize = $true

$numSpeed = New-Object System.Windows.Forms.NumericUpDown
$numSpeed.Minimum = 0.5; $numSpeed.Maximum = 10; $numSpeed.Increment = 0.5; $numSpeed.Value = 1
$numSpeed.Location = New-Object System.Drawing.Point(80, 68)
$numSpeed.Size = New-Object System.Drawing.Size(70, 24)

$cbPacing = New-Object System.Windows.Forms.ComboBox
$cbPacing.DropDownStyle = "DropDownList"
[void]$cbPacing.Items.AddRange(@("0（最速）", "500（通常）", "1500（観戦）", "3000（学習）"))
$cbPacing.SelectedIndex = 2
$cbPacing.Text = "1500（観戦）"
$cbPacing.Location = New-Object System.Drawing.Point(330, 68)
$cbPacing.Size = New-Object System.Drawing.Size(150, 24)

$numRuns = New-Object System.Windows.Forms.NumericUpDown
$numRuns.Minimum = 1; $numRuns.Maximum = 20; $numRuns.Value = 1
$numRuns.Location = New-Object System.Drawing.Point(610, 68)
$numRuns.Size = New-Object System.Drawing.Size(60, 24)

$gbSet.Controls.AddRange(@(
    (New-Label "キャラ" 14 30), $cbChar,
    (New-Label "アセンション" 240 30 90), $numAsc,
    (New-Label "シード" 420 30 60), $txtSeed, $chkRandom,
    (New-Label "速度" 14 70), $numSpeed,
    (New-Label "間隔(ms)" 240 70 90), $cbPacing,
    (New-Label "ラン数" 550 70 60), $numRuns
))
$lblHint = New-Object System.Windows.Forms.Label
$lblHint.Text = "Mentor: 間隔は観戦向け（自動）。独自 MOD: 速度と間隔を反映。"
$lblHint.Location = New-Object System.Drawing.Point(14, 110)
$lblHint.AutoSize = $true
$lblHint.ForeColor = [System.Drawing.Color]::DimGray
$gbSet.Controls.Add($lblHint)
$form.Controls.Add($gbSet)

# ---- ボタン
$btnStart = New-Object System.Windows.Forms.Button
$btnStart.Text = "起動"; $btnStart.Location = New-Object System.Drawing.Point(14, 250)
$btnStart.Size = New-Object System.Drawing.Size(150, 40)
$btnStop = New-Object System.Windows.Forms.Button
$btnStop.Text = "停止"; $btnStop.Location = New-Object System.Drawing.Point(180, 250)
$btnStop.Size = New-Object System.Drawing.Size(110, 40)
$btnStatus = New-Object System.Windows.Forms.Button
$btnStatus.Text = "状態"; $btnStatus.Location = New-Object System.Drawing.Point(300, 250)
$btnStatus.Size = New-Object System.Drawing.Size(110, 40)
$btnPause = New-Object System.Windows.Forms.Button
$btnPause.Text = "一時停止/再開"; $btnPause.Location = New-Object System.Drawing.Point(420, 250)
$btnPause.Size = New-Object System.Drawing.Size(120, 40)
$btnManual = New-Object System.Windows.Forms.Button
$btnManual.Text = "手動（推奨のみ）"; $btnManual.Location = New-Object System.Drawing.Point(548, 250)
$btnManual.Size = New-Object System.Drawing.Size(130, 40)
$btnStep = New-Object System.Windows.Forms.Button
$btnStep.Text = "1 手"; $btnStep.Location = New-Object System.Drawing.Point(690, 250)
$btnStep.Size = New-Object System.Drawing.Size(70, 40)
$form.Controls.AddRange(@($btnStart, $btnStop, $btnStatus, $btnPause, $btnStep, $btnManual))

# ---- 出力
$txtOut = New-Object System.Windows.Forms.TextBox
$txtOut.Multiline = $true
$txtOut.ScrollBars = "Vertical"
$txtOut.ReadOnly = $true
$txtOut.Location = New-Object System.Drawing.Point(14, 300)
$txtOut.Size = New-Object System.Drawing.Size(710, 270)
$txtOut.Font = New-Object System.Drawing.Font("Consolas", 9)
$form.Controls.Add($txtOut)

function Log($text) {
    $txtOut.AppendText("[" + (Get-Date -Format "HH:mm:ss") + "] " + $text + "`r`n")
}

function Invoke-Wsl([string[]]$cmdArgs) {
    # wsl.exe を引数配列で呼ぶ（パスの空白に強い）
    $all = @("--cd", $linuxRepo, "--", "bash", "-lc") + $cmdArgs
    return & wsl.exe @all 2>&1
}

function Get-MentorSeed {
    if ($chkRandom.Checked -or [string]::IsNullOrWhiteSpace($txtSeed.Text)) { return "" }
    return $txtSeed.Text.Trim()
}

$btnStart.Add_Click({
    $seed = Get-MentorSeed
    $pacing = ($cbPacing.SelectedItem -replace '\D.*$', '')

    if ($rbNormal.Checked) {
        Log "通常プレイで起動します"
        Start-Process -WindowStyle Hidden -FilePath "wsl.exe" -ArgumentList @("--cd", $linuxRepo, "--", "bash", "-lc", "./play.sh launch normal")
        return
    }
    if ($rbAI.Checked) {
        $cmd = "./play.sh launch ai --character '$($cbChar.SelectedItem)' --speed $($numSpeed.Value) --pacing $pacing"
        if ($seed) { $cmd += " --seed $seed" }
        Log ("AI 観戦で起動: " + $cmd)
        Start-Process -WindowStyle Hidden -FilePath "wsl.exe" -ArgumentList @("--cd", $linuxRepo, "--", "bash", "-lc", "$cmd")
        return
    }
    # Mentor
    $cmd = "./play.sh mentor --character '$($cbChar.SelectedItem)' --ascension $($numAsc.Value) --runs $($numRuns.Value)"
    if ($seed) { $cmd += " --seed $seed" }
    Log ("Mentor 観戦で起動: " + $cmd)
    Start-Process -WindowStyle Hidden -FilePath "wsl.exe" -ArgumentList @("--cd", $linuxRepo, "--", "bash", "-lc", "$cmd")
    Log "ウィンドウが出たら、ゲーム画面を確認してください。"
})

$btnStop.Add_Click({
    Log "停止します"
    Start-Process -WindowStyle Hidden -FilePath "wsl.exe" -ArgumentList @("--cd", $linuxRepo, "--", "bash", "-lc", "./play.sh launch --stop >/dev/null 2>&1; pkill -f 'local mentor'; pkill -f 'local autopilot'; true")
    Log "停止を要求しました（ゲームは閉じません）"
})

$btnStatus.Add_Click({
    Log "--- 状態 ---"
    $s = Invoke-Wsl @("./play.sh mentor --status 2>/dev/null; ./play.sh launch --status 2>/dev/null")
    Log (($s -join "`r`n"))
})

$btnPause.Add_Click({
    $r = Invoke-Wsl @("./play.sh launch --pause")
    Log (($r -join "`r`n"))
})

$btnStep.Add_Click({
    $r = Invoke-Wsl @("./play.sh launch --step")
    Log (($r -join "`r`n"))
})

$btnManual.Add_Click({
    $r = Invoke-Wsl @("./play.sh launch --manual")
    Log (($r -join "`r`n"))
})

Log "リポジトリ（WSL）: $linuxRepo"
Log "起動ボタンを押すと、ゲームが立ち上がります。"

[void]$form.ShowDialog()
