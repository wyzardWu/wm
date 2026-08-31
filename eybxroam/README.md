---
# TODO: 选一个许可证再填，比如 cc-by-4.0 / cc-by-nc-4.0 / other
# license: cc-by-nc-4.0
language:
- zh
tags:
- video
- world-model
- game
- no-rest-for-the-wicked
- open-world
- navigation
- action-conditioned
size_categories:
- 1K<n<10K
---

# No Rest For The Wicked · 开放世界漫游 — 视频 + 同步状态数据集

一个 BOT 在《No Rest For The Wicked》的世界地图上连续漫游的录像，配套 **约 20 Hz 的玩家状态**
和**逐帧的键鼠输入**，所有日志和视频共用同一个时间零点。用于训练 action-conditioned 的视频世界模型。

**2 轮连续录制，50.9 小时，149 GB 视频（1280×720 @30 fps），1604 段漫游，走了 327 km，52 个区域。**

| session | 时长 | 视频 | 玩家采样 | 段数 | 传送 | 里程 | 区域 | **有效数据** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **`20260821_190601_787`** | 34.55 h | 96.80 GB | 2,300,504 | 1015 | 1067 | 216.0 km | 50 | **27.32 h（79.3 %）** |
| **`20260823_201942_753`** | 16.39 h | 52.17 GB | 1,162,052 | 589 | 650 | 111.5 km | 39 | **13.31 h（82.0 %）** |
| **合计** | **50.94 h** | **148.97 GB** | **3,462,556** | **1604** | **1717** | **327.5 km** | **52（并集）** | **40.63 h（80.2 %）** |

**动作空间只有 9 个离散动作**：8 个移动方向（W / WA / A / AS / S / SD / D / DW，`mv.deg` 取
2 / 46 / 92 / 136 / 180 / 226 / 270 / 316）加上「不动」，`mv.mag` 恒为 **1.0**。
两轮里八个方向的分布都很均匀（各 9~17 %），有移动输入的帧占 91~92 %。

**⚠️ 用之前必须先读第 1 节。** 两轮的**环境和音频不一样**，直接混会给模型灌进一个它学不到的变量。

---

## 0. 该用哪一份

- **想要环境恒定、画面干净的** → `20260823_201942_753`。昼夜钉死在 Day、天气钉死在 Clear、
  全程静音，可用率也更高（82.0 %）。**做视频世界模型推荐这一份。**
- **想要昼夜和天气变化的** → `20260821_190601_787`。它没有环境钉死，游戏自己的昼夜循环和天气
  在跑，视频里也有游戏原声。时长是前者的两倍多。
- **两份都要** → 可以，但请把「环境是否恒定」当成一个已知的域差异处理，见下一节。

---

## 1. 两轮的差别（必须先读）

| | `20260821_190601_787` | `20260823_201942_753` |
|---|---|---|
| **昼夜 / 天气** | **随游戏自己变化**（有白天黑夜、有雨） | **钉死 Day / Clear**（968 条 `env_freeze` 事件为证） |
| **音频** | **有游戏原声**（AAC 192 kb/s，均值 −34 dB） | **静音**（AAC 2 kb/s，均值 −91 dB，即数字静音） |
| **日志损坏** | 无 | **有：vt 5118 → 5655 共 537 秒**，见第 4 节 |
| 到过的传送点 | 59 个 | 46 个（坏点黑名单变长了，见 4.2） |
| 采样率 | 18.50 Hz | 19.69 Hz |
| 段结束原因 | 原地打转 555 / 卡住 199 / 走完 154 / 被游戏挪走 51 / 路线走完 24 / 无处可走 23 / 掉下悬崖 8 | 原地打转 384 / 卡住 90 / 被游戏挪走 60 / 走完 29 / 路线走完 15 / 无处可走 10 |
| 高度范围 y | −259 ~ 132（有掉进深坑的段） | −1 ~ 133 |
| 怎么结束的 | 跑了 34.5 小时后**被游戏自己终止**（音频资源耗尽） | **手动停止**，没有崩 |

**能不能混，取决于你的任务：**

- **短片段（几秒级）的动力学、导航、碰撞 —— 可以直接混。** 两轮的地图、物理、动作空间、
  采样率、分辨率完全一致，短窗口里发生的事没有区别。
- **要求画面分布一致的任务（纯视频预测、对光照/天气敏感的表征）—— 分开，或者只用 `20260823`。**
  一个是恒定正午晴天，一个横跨昼夜和降雨；混在一起等于给模型加了一个它无法从动作里推断的隐变量。
- **要音频的 —— 只有 `20260821` 有。** `20260823` 的音轨是全静音的占位轨，不是丢了。

---

## 2. 怎么对齐

**一句话：所有日志里的 `vt` 就是视频的第几秒，零点是 OBS 开始录制的那一刻。**

`boss深度日志.jsonl` 第一行 `{"type":"meta", ...}` 里有 `videoOriginMs`（录制起点的 Unix 毫秒）。
任意一行的 `vt = (w - videoOriginMs) / 1000`，`w` 是该采样的 UTC 毫秒。
所有文件里的 `视频时间` 字段是同一个零点，格式 `HH:MM:SS:FF`（FF 是 30 fps 下的帧号）。

取某一段漫游的画面：

```python
import json, io

path = "20260823_201942_753/boss深度日志.jsonl"
ev = []
for line in io.open(path, encoding="utf-8", errors="replace"):
    if '"type":"event"' not in line:
        continue
    try:
        ev.append(json.loads(line))
    except Exception:
        pass                      # 见 4.1：有一段是损坏的

starts = {e["leg"]: e for e in ev if e["ev"] == "roam_leg_start"}
ends   = {e["leg"]: e for e in ev if e["ev"] == "roam_leg_end"}

leg = 42
a, b = starts[leg], ends[leg]
print(a["spawnIndex"], b["reason"], b["seconds"], "→ 视频", a["vt"], "到", b["vt"])
```

```bash
ffmpeg -ss 1234.5 -to 1354.5 -i 20260823_201942_753/video.mp4 leg42.mp4
```

**按场景切片**用 `场景外观日志.jsonl` 更省事 —— 它是稀疏的（1087 行 / 601 行），一行一次传送，
不用去几百 MB 的日志里捞。

---

## 3. 文件与字段

每个 session 目录：

| 文件 | 说明 |
|---|---|
| `video.mp4` | 1280×720 @30 fps，H.264 |
| `boss深度日志.jsonl` | 逐帧状态 + 事件。**主文件。** 名字沿用打 BOSS 那套，漫游里也是它 |
| `输入日志.jsonl` | 逐帧键鼠输入，和上面同一个时间轴 |
| `输入日志.json` | 人可读的按键事件流（按下/松开），给人看的，不是训练用的 |
| `场景外观日志.jsonl` | 稀疏的换场景记录，一行一次传送 |
| `session.json` | 录制元信息（OBS 起止 UTC、原始路径） |

### `boss深度日志.jsonl`

第一行是 `meta`，之后每行一个 `s`（状态采样）或 `event`（事件）。

**`type: "s"` — 状态采样**（约 20 Hz。漫游时基本只有 `e:"player"`；`e:"boss"` 各只有几十条，是偶尔锁到的目标）

| 字段 | 含义 |
|---|---|
| `w` | UTC 毫秒 |
| `vt` | 视频秒（对齐用这个） |
| `视频时间` | `HH:MM:SS:FF` |
| `t` | 游戏秒 = `(f − 首帧f) / 60`。Quantum 固定 60 Hz，比墙钟准 |
| `f` | Quantum 模拟帧号 |
| `e` | 实体类型：`player` / `boss` |
| `id` | 实体 ID，`Index:Version` |
| `hp` / `hpm` | 当前 / 最大血量。**漫游数据里恒为 100，玩家全程没掉过血** |
| `sp` | 当前所属传送点序号 |
| `rg` | **区域名**（如 `sacramentA`、`sewerDungeonA`）。切场景用它 |
| `x` / `y` / `z` | 世界坐标，米。`y` 是高度 |
| `yaw` | 朝向 |

**`type: "event"` — 事件**

| `ev` | 含义 | 独有字段 |
|---|---|---|
| `roam_start` | 漫游开始 | `points`（传送点表大小，276） |
| `roam_scene` | 换场景（发出一次传送） | `leg` `sp` `x` `z` `rg` `ord` `total` |
| `roam_leg_start` | 一段开始 | `leg` `spawnIndex` `x` `z` `nav`（有没有导航网格）`reachablePolys` |
| `roam_leg_end` | 一段结束 | `leg` `spawnIndex` `reason` `seconds` `pathMeters` `netMeters` `straightness` `avoided` `replans` |
| `roam_leg_failed` | 这一段没走成 | — |
| `spawn_rejected` | 传送点被拒 | — |
| `env_freeze` | 昼夜/天气被钉住（**只有 `20260823` 有**） | `time` `weather` `fixes` |
| `roam_stop` / `session_stop` | 收尾 | — |

`reason` 的取值：`原地打转` / `卡住` / `走完` / `被游戏挪走` / `路线走完` / `无处可走` / `掉下悬崖` / `停机` / `人工淘汰`。

### `输入日志.jsonl`

| 字段 | 含义 |
|---|---|
| `w` `vt` `视频时间` `t` `f` | 同上，同一时间轴 |
| `p` | 本地玩家序号（恒为 0） |
| `down` / `pressed` / `released` | **游戏动作名**（如 `Run`），**不含移动** |
| `mv` | **移动向量。动作空间在这里。** `deg` 方向角、`mag` 幅值（恒 1.0）、`keys` 按键组合（`W` / `WA` / …）、`rawAngle` / `rawMag` 是定点原值 |
| `aim` | 瞄准方向。**漫游数据里基本是常量**（95 % 落在同一个方向），因为漫游 BOT 不攻击 —— 别拿它当特征 |
| `camRot` | 相机旋转原值 |
| `scroll` / `quickItem` | 滚轮 / 快捷物品 |

**动作空间就是 `mv.keys`（或等价的 `mv.deg`）的 9 个取值**：空（不动）+ 8 个方向。
`mv.mag` 恒为 1.0，没有模拟量。`Run`（冲刺）在 `20260823` 里只出现 86 帧（0.0 %），
在 `20260821` 里一次都没有 —— **两轮都是走路速度**。

---

## 4. 已知问题（用之前要处理的）

### 4.1 `20260823` 有 537 秒日志损坏 —— 必须跳过

`vt 5118.03 → 5655.37`（视频 `01:25:18` → `01:34:15`）这一段，深度日志和输入日志都写成了
**均匀随机字节**（熵 8.000 bits/byte）。**视频那 9 分钟是好的，日志回不来了。**

它伪装性很强：**坏字节数正好等于那段本该写的量**，所以文件大小、总行数、能不能打开全都正常。
按行 `json.loads` 时它们会抛异常，try/except 跳过即可 —— 占深度日志 0.89 %、输入日志 0.91 %：

```python
import json, io
good, bad = [], 0
for line in io.open(path, encoding="utf-8", errors="replace"):
    try:
        good.append(json.loads(line))
    except Exception:
        bad += 1
```

`20260821` 的 1.2 GB 日志**零损坏**（只有最后一行被截断，那是游戏被强制终止留下的）。

### 4.2 `场景外观日志` 里的 `"total":66` 是误导

`total` 是路线长度，不是**实际会去的点数**。路线上有一部分点在坏点黑名单里（传过去到不了、
或落地是四周无路的孤岛），BOT 会直接跳过，所以：

- `20260821` 实到 **59** 个点
- `20260823` 实到 **46** 个点（黑名单在这两天里变长了）

**算覆盖率请用实际出现过的不同 `ord` / `sp` 数，不要用 `total`。**

### 4.3 传送那一拍是位置突变

相邻两个采样之间位移 > 20 m 就是一次传送（不是走出来的）。做轨迹、速度、动力学时**必须在这里断开**，
否则会得到一个几千米每秒的样本。全数据集共 1717 次。

### 4.4 有 8~12 % 的时间是「有输入但没位移」

BOT 撞墙 / 卡地形。已经在上表的「有效数据」里扣掉了。判据是**1 秒窗口内位移 < 0.5 m 且发了移动指令**
—— 不能用单拍位移，20 Hz 下正常步速才 0.12 m/拍，比噪声大不了多少。

### 4.5 其它

- `hp` 恒为 100：漫游 BOT 空手、场景里的怪被清掉了，全程没受过伤 —— **这份数据里没有战斗**。
- `aim` 基本是常量，见第 3 节。
- `20260821` 另有 162 处零星断档，合计 6.7 分钟，最长 5.2 秒 —— 正常抖动，不是损坏。

---

## 5. 采集是怎么做的

- 游戏本体 Unity 6000.1.15f1 + IL2CPP，物理/状态走 Photon Quantum 确定性 ECS，模拟 60 Hz。
  **状态是从模拟帧里直读的，不是从画面上认的。**
- BOT 通过 Win32 `SendInput` 合成真实键鼠事件 —— 走的是和人完全同一条输入通路，
  所以 `mv` 才会是干净的 8 方向离散值。
- 一段漫游 = 传送到一个传送点 → 沿导航网格朝可达区域走 → 直到超时 / 卡住 / 无路可走 → 传下一个点。
- 录制端 OBS 由插件通过 WebSocket 驱动，`videoOriginMs` 就是 OBS 真正开始写文件的时刻。
- 场景里的怪会被清掉、刷怪器会被关掉 —— 目的是让画面里只剩地形和光照，减少无关变量。
- `20260823` 额外把昼夜钉在 Day、天气钉在 Clear，并把整个 Wwise 声音引擎挂起。

## 6. 走到了哪些地方

- 坐标范围 x −4129 ~ 3088、z −2580 ~ 1320、y −259 ~ 133
- 按 100 m 栅格算，两轮合计走过 **120 格 ≈ 1.20 km²**（`20260821` 114 格，`20260823` 87 格）
- 区域并集 **52 个**，停留最久的是 `sacramentA`（城区）、`sewerDungeonA`（下水道）、
  `mountainPassA`（山道）、`willsFarmA`（农场）
- 传送点并集 **86 个**

**注意：两轮的空间覆盖高度重叠** —— `20260823` 只带来 6 个新格子、2 个新区域。
如果你需要的是空间多样性而不是时长，这两轮加起来的边际收益不大。

---

## English summary

Continuous open-world roaming in *No Rest For The Wicked*, recorded as video plus time-aligned
state and input logs, for training action-conditioned video world models.

**2 sessions, 50.9 hours, 149 GB of 1280×720@30fps video, 1604 roaming legs, 327 km travelled,
52 distinct regions.** Player state is sampled at ~20 Hz directly from the game's deterministic
Quantum simulation (not inferred from pixels); inputs are logged per frame off the same clock.
All logs share one time origin with the video: the `vt` field is literally "seconds into the video".

The action space is **9 discrete actions**: 8 movement directions (WASD + diagonals) plus idle.
`mv.mag` is always exactly 1.0 — there is no analog movement, and no sprinting.

**Read section 1 before mixing the two sessions.** `20260821_190601_787` has the game's own
day/night cycle, weather, and game audio. `20260823_201942_753` has time frozen at Day, weather
frozen at Clear, and a fully silent audio track. Short-horizon dynamics can be mixed freely;
anything sensitive to the image distribution should not.

`20260823_201942_753` has a **537-second corrupted window** at `vt 5118–5655` (section 4.1) —
those log lines are uniform random bytes and must be skipped. The video for that window is fine.
