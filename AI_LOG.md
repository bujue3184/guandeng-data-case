# AI_LOG

记录与 AI 协作的关键交互：提示词、AI 出错的地方、我如何纠正。
环境：Windows 11 + WorkBuddy(agent 模式) + Python 3.13 + pandas。

---

## A. 开赛前：环境与交付链路准备

### A1. 任务链接探测

**我的提示词**
> 测试当前机器环境，网络环境，浏览器控制情况，准备执行此任务。

**AI 做了什么**
用 `curl` 探测任务链接，返回 HTTP 404；随后连续多次超时（HTTP 000），AI 初步判断为「网络受阻」。

**AI 哪里不对**
把「连接失败」和「链接无效」混为一谈。实际对比测试后确认：域名 `app-staging-eace.up.railway.app` 基础路径返回 302（可达），只有 `/t/<token>` 路径稳定返回 404 —— 也就是**服务端说链接无效或过期，不是网络问题**。AI 后续用真实浏览器打开页面，抓到页面正文「链接无效或已过期」，才修正结论。

**我的纠正**
要求「不要靠单一 curl 结果下结论，用真实浏览器交叉验证，并区分 4xx 与连接层错误」。

### A2. 浏览器控制可用性

**AI 哪里不对**
`agent-browser` 打开站点时报 `net::ERR_TUNNEL_CONNECTION_FAILED`，AI 第一反应是网络故障。

**纠正过程**
排查发现本机存在环境变量级代理 `HTTP(S)_PROXY=127.0.0.1:59839`，Chromium 走系统代理隧道失败。解决：重启浏览器守护进程并加 `--no-proxy-server` 直连，页面即可正常加载。
**经验**：浏览器自动化报隧道错误时，先查 `HTTP(S)_PROXY` 与系统代理，而不是怀疑目标站点。

### A3. GitHub 推送授权（本次最大的弯路）

**AI 哪里不对（重要）**
AI 选择 Git Credential Manager(GCM) 交互式登录，连续尝试三次：
1. `git credential fill` 挂起 25 分钟才返回（拿到一个很快过期的 token）；
2. 切换 `credentialStore=dpapi` 后重试，用户明确反馈「**未收到任何类型的弹窗**」；
3. 又尝试 `GCM_GITHUB_AUTHMODES=device` 设备码流，日志始终为空、进程挂起。

**根因**
该运行环境（沙箱/便携 Git）无法把 GCM 的 GUI 窗口投递到用户桌面会话，交互式授权链路整体不可用。AI 在没有确认「窗口是否真的可见」之前，重复了同一种失败策略，浪费了约 30 分钟。

**我的纠正**
停止重试 GUI 授权，改为在 `~/.gitconfig` 中为 github.com 单独配置凭据存储与代理：
```ini
[http "https://github.com/"]
    proxy = http://127.0.0.1:7897
[credential "https://github.com"]
    helper =
    helper = manager
    credentialStore = wincredman
```
之后 `git credential fill` 秒回，clone / push / pull 全部通过（已验证：新建私有测试仓库 → 提交 → push → 重新 clone 校验内容一致 → API 返回 200）。

**可复用的教训**
> 在无人值守环境里，凡是「必须人工点弹窗」的授权方案，都应当先验证窗口能否真正到达用户眼前；不能，就立刻换非交互方案（token / 环境预置凭据），不要在同一个失败点反复试探。

### A4. 分析环境预置

- 建立独立 venv（隔离，不污染系统 Python），安装 pandas / numpy / matplotlib / openpyxl。
- 预先写好 `scripts/profile.py`：未知 CSV 的自动体检（编码与分隔符自适应、字段画像、候选主键、时间跨度、类别分布），保证拿到数据后第一时间能出结构认知，而不是把 30 分钟花在读表上。

---

## B. 正式分析阶段

> _（任务开始后按实际交互补齐）_

| # | 我的提示词 | AI 输出/判断 | 问题 | 我的处理 |
|---|---|---|---|---|
| B1 | | | | |
