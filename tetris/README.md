# xDecision 玩俄罗斯方块

让决策模型 **xDecision** 在浏览器里自动玩 [vanilla-js-tetris](https://github.com/trzaskos/vanilla-js-tetris)。
模型通过 webtorch SDK 在本机显卡上运行，每来一个方块，就由模型决定把它放在哪里。操作游戏时直接调用
游戏自己的函数，不模拟键盘，也不读屏幕像素。

```bash
node serve-coi.mjs . 8119
# 打开 http://localhost:8119/tetris/
```

部署到 Pages 后的地址是 `https://xnetsc.github.io/webpytorch/tetris/`。

## 用的是哪个模型，从哪里下载

就是 Pages 聊天应用里列出的那个 xDecision（`chat/models.json`）：仓库 `mccoysc/xDecision`，文件
`models/gguf/xDecision-Q8_0.gguf`（402,546,752 字节）。页面上只有一个“加载”按钮，模型和下载源都不用选。

下载和缓存都用 SDK 现成的接口，页面自己不写这套逻辑：

- **`webtorch.mirrored_read()`**：同时从 Hugging Face、ModelScope .cn、ModelScope .ai 读，每一块都从当时
  答得最快的那个站取，某个站变慢或断了就换下一个。混着读之前先核对各站的文件是同一个（优先比各站公布的
  哈希，拿不到就比长度和抽样的几块），没有这个文件的站自然被排除。
- **缓存**：读取器默认把下载的块存进浏览器（IndexedDB），中断了下次接着下；页面再装上
  `set_io_write(webtorch.default_io_write)`。下完之后再打开页面，模型直接从缓存读，按钮会显示
  “从缓存加载 xDecision”（用 `wt.cache.list()` 判断缓存里有没有完整的一份）。

```js
const wt = await webtorch.start({ baseURL: '../' });
await wt.run(`import webtorch
webtorch.set_io_read(webtorch.mirrored_read())
webtorch.set_io_write(webtorch.default_io_write)`);
const info = await wt.load('mccoysc/xDecision/models/gguf/xDecision-Q8_0.gguf');
const res  = await wt.decide(state, { move: question });   // res.answers.move.choice
```

加载时页面显示 SDK 报告的两种进度：读了多少字节（已读取 / 总大小，下载时还有下载速度，从缓存读时注明来自缓存），
以及当前在做哪一步（读取权重并放到显卡上，带张量计数；预热；在这块显卡上测速；为每种输入长度录制计算过程），
加载完成后列出各步用时。读取完成以后后面几步还要一段时间，所以第二行是为了让人知道它没有卡住。

手里已经有这个文件的话，也可以从自己的服务器加载：`/tetris/?model=/models/gguf/xDecision-Q8_0.gguf`
（同源，或者对方允许跨域）。

## 每一步怎么决定

xDecision 根据给定的事实回答问题，不会推理，可以把它当一个便宜的 if-else 用。所以交给它的局面要把做决定
需要知道的东西都用文字写出来：这个游戏怎么玩、怎么算赢、怎么算输、要避免什么，当前是什么方块，每个候选落点
放下去会怎样、这对输赢意味着什么。问题只问“哪个落点最有机会赢”，选项就是这几个落点。

每来一个新方块，`ai.js` 做四件事：

1. **读局面。** 通过 `bridge.js` 拿到游戏自己的 `board`（10×20）、当前方块 `player.matrix` / `player.pos`，
   以及预览里的下一个方块 `player.next`。
2. **列出所有落点。** `planner.js` 按游戏的操作顺序（先旋转若干次，再左右移动，最后直接落下）找出每一个
   能到达的落点。旋转用的就是游戏自己的 `rotate()`，碰撞检测和旋转时的踢墙逐行照抄 `tetris.js`，所以算出来的
   结果和游戏实际一致。然后按 Dellacherie 特征（权重用 El-Tetris 公开的数值）加上对下一个方块的一步预判排序。
3. **用文字写出局面，问 xDecision。** 每个落点放下去的结果看两条：消不消行，留不留新洞。从前 K 名（默认 4 名）
   里，每一类结果只留排在最前的那个落点，所以候选最多 4 个，结果各不相同。交给模型的是：

   ```json
   {
     "state": {
       "game": "Tetris",
       "how_to_play": "Pieces fall one at a time; you choose where each lands. A completely filled row disappears: clearing lines is how you win.",
       "how_you_lose": "You lose when the stack reaches the top.",
       "what_to_avoid": "A hole (an empty cell covered by a block) cannot be cleared and pushes the stack up. Avoid holes.",
       "piece": "T",
       "placements": [
         {"id": "A", "result": "hole-free", "effect": "no hole: keeps you safe"},
         {"id": "B", "result": "clears a line, hole-free", "effect": "clears a line with no hole: helps you win"},
         {"id": "C", "result": "makes a new hole", "effect": "makes a hole: helps you lose"}
       ]
     },
     "questions": {"move": {
       "type": "choice",
       "instructions": "Which placement gives the best chance to win?",
       "criteria": {"A": null, "B": null, "C": null}
     }}
   }
   ```

   `id` 就是棋盘上标的 A、B、C、D；`result` 是放下去会怎样，`effect` 是按上面的原则，这对输赢意味着什么。
   选项就是这几个字母，尽量短：问题和选项共用模型输入开头的 256 个 token，整个输入最多 1024 个 token，
   现在每步 170–250 个左右。模型选概率最高的那个字母，代码再把字母对回落点。题型名不是写死的，加载后从模型的
   `surface()` 里找“由调用方给选项”（shape 为 `named`）的那一种。

   落点具体在哪几列、哪几行，页面自己知道就够了，没有写进局面：实测加上这些（以及棋盘高度、下一个方块）以后，
   模型反而选得更差（见下面的表）。这样模型的输入只取决于两件事：出现了哪几类结果（11 种组合）和当前方块
   （7 种），一共 77 个可能的输入。`tetris/eval/table.mjs` 把这 77 个全部问过，模型都选了最有利的那个（有不留洞
   的消行就选它；没有，就选不留洞的；再没有，就选能消行的），所以实际对局里它不会遇到没验证过的输入。候选按
   固定的顺序列出，正确答案有时排第一、有时第二、有时第三，不是靠位置。
4. **执行。** 用游戏自己的 `playerRotate()`、`playerMove()`、`playerHardDrop()` 把方块放过去。执行前按
   落点再找一遍路径，模型思考期间方块如果往下掉了一格，也还是落到模型选的位置。

页面上会列出每一步的选项、模型给每个选项的概率、模型选了哪个（★），以及按同样的原则在代码里判断出的答案（✓），
可以核对模型判断得对不对。“代码”模式不调用模型，直接用代码判断的答案，用来对比。

## 怎么调用游戏的内部接口

`vendor/tetris.js` 原样拷贝自上游（提交 `f09e736`），一个字都没改。它是普通的 `<script>`，没有导出任何东西，
但这不妨碍调用：

- 它顶层的 `function`（`playerRotate`、`playerMove`、`playerHardDrop`、`rotate`、`resetGame`、`togglePause`）
  挂在全局对象上；
- 它顶层的 `let` / `const`（`board`、`player`、`score`、`lines`、`level`、`gameOver`、`isPaused`、
  `dropCounter`）在所有普通脚本共享的全局词法作用域里，后加载的脚本可以直接按名字读写，只是不在 `window` 上。

`bridge.js` 在它之后加载，是唯一碰这些内部变量的文件：

| bridge.js | 调用 / 读取的游戏内部 |
|---|---|
| `snapshot()` | `board`、`player.matrix`、`player.pos`、`player.next`、`score`、`lines`、`level`、`gameOver`、`isPaused`（复制一份） |
| `token()` | `player.matrix` 的引用。`playerReset()` 每来一个新方块都换一个新数组，旋转只在原数组上改，所以引用一变就是来了新方块 |
| `rotateMatrix()` | `rotate()`，给 planner 转副本用 |
| `pressRotate()` / `pressMove()` / `pressDrop()` | `playerRotate()` / `playerMove()` / `playerHardDrop()`，和键盘处理函数调的一模一样 |
| `perform(plan)` / `step(plan)` | 同上三个函数，按计划一次走完，或者一次按一下 |
| `holdGravity(on)` | 把 `dropCounter` 设成 `-Infinity`：游戏循环 `update()` 就不会让方块下落，也不会出现 `togglePause()` 的 PAUSED 遮罩。`playerHardDrop()` 会自己把它清零 |
| `reset()` / `pause()` | `resetGame()` / `togglePause()` |

另外两处小改动在页面里，不在游戏文件里：

- 画布宽 200px（10 列 × 20px）。上游是 240px，这就是它 README 里写的“方块到不了最右边”的已知问题：棋盘只有 10 列，画布却画了 12 列宽。
- `playerMove()` 每调用一次都会 `console.log` 一行（上游留下的调试输出）。AI 移动时只在调用期间临时屏蔽
  `console.log`。

## 实测效果

### 整局

用 `eval/eval.mjs` 测：在 Node 里跑原版 `tetris.js`，种子 1–5 各一局，每局最多 500 块。每一步给出的选项完全相同，
只换“谁来选”：

| 谁来选 | 平均消行 | 平均放下方块 | 500 块内输掉 | 和代码判断的答案一致 |
|---|---:|---:|---:|---:|
| **xDecision**（页面默认） | 198.6 | 500 | 0 / 5 | 100% |
| 按同样的原则在代码里判断（“代码”模式） | 198.6 | 500 | 0 / 5 | 100% |
| 规划器排第一的 | 198.4 | 500 | 0 / 5 | 86.7% |
| 在同样的选项里随机选 | 25.6 | 102.2 | 5 / 5 | 56.2% |

在没有显卡的机器上，用 SDK 的 CPU 路径（numpy）跑，每步约 220 毫秒。浏览器里走 WebGPU，权重相同，但数值
路径不同，所以 77 个输入的检查是针对 CPU 路径做的。

### 这个模型能做什么、不能做什么

这些结果决定了上面的设计。“77”是全部可能的输入；“真实局面”取自真实对局（120 个，后三行是 240 个）；“1848”是给位置名
专门做的测试集（77 种情况 × 24 种名字分法）：

| 问法 | 结果 |
|---|---|
| **局面写原则 + 每个候选的 `result` 和 `effect`，问“哪个落点最有机会赢”，选项是字母（现在用的）** | **77 / 77；真实局面 120 / 120** |
| 同上，但不写 `effect`（只有原则和 `result`） | 42 / 77 |
| 同上，问题换成“下一步该放哪个”（“Which placement should be played next?”） | 50 / 77 |
| 再加上棋盘概况 / 下一个方块 / 每个落点在哪几列哪几行 | 真实局面 111 / 85 / 89（共 120） |
| 候选不按结果分类，直接给规划器的前 4 名 / 前 8 名 | 真实局面 104 / 74（共 120） |
| 原则和局面写成整段文字，而不是一条条记录 | 约 65%，大多选 A |
| 规则直接写成题目（“选既消行又不留洞的”），局面里只有标签（上一版） | 77 / 77，但这只是让模型对标签，不是决策 |
| 选项是落点位置（`columns 2-3, rows 1-3`），不是字母 | 测试集 1848 / 1848，但真实对局只有 99.5%：位置名由相同的数字组成时（`columns 2-3, rows 1-3` 和 `columns 1-3, rows 2-3`），两个落点的结果会被弄混 |
| 没有选项符合时，答“都不是”（`none`） | 3% 答对：几乎从不选 `none` |
| 结果在局面里，按字母问“选项 B 不留洞”（是/否题） | 所有问题都答“是” |
| 把一句事实分到几个只差一个“不”字的类别里 | 51% 答对 |
| 每个选项写一串相对比较（“落点最低、最平、边缘最整齐…”），让它挑最好的 | 63% 选中规划器第一名，整局平均 106 行、300 块就输 |
| 选项写成数字（`stack 8 high, bumpiness 10`） | 240 次里 218 次选最后一个选项，跟内容无关 |

所以：局面里写清楚怎么玩、怎么输、要避免什么，再给每个候选写上它放下去的结果和这对输赢意味着什么。只有原则
不够，模型不会自己把“不留洞”和“要避免洞”连起来，`effect` 这一行就是把这一步写出来。局面里只放做决定要用的
东西，棋盘写得越细，它反而越容易选错。选项用字母，又短，又不会和结果混在一起；不设“都不是”；同类结果只留一个候选。

### 浏览器

完整流程在无界面 Chromium 里跑过：通过 SDK 加载 xDecision、模型连续决策并落子、候选标在棋盘上、“代码”模式、
游戏结束后自动重开、停止后重力恢复、手机宽度没有横向滚动。那台机器没有显卡，WebGPU 是 CPU 模拟的
（`swiftshader`，`isFallbackAdapter: true`），每步约 18–21 秒（模型每步读 170–250 个 token）；在真正的显卡上会快得多，但没有在显卡上实测过这个页面。

## 文件

| | |
|---|---|
| `index.html` | 页面；保留了 `tetris.js` 要找的元素 id（`tetris`、`next`、`score`、`lines`、`level`、`start-button`） |
| `vendor/tetris.js` | 上游游戏，原样 |
| `bridge.js` | 游戏内部接口 → `window.TetrisGame` |
| `planner.js` | 落点枚举、局面特征，以及交给模型的局面和问题（浏览器和 Node 都能用） |
| `ai.js` | SDK 启动、模型加载、每一步的决策和执行、界面 |
| `eval/headless.mjs` | 在 Node 的 vm 里跑原版 `tetris.js` + `bridge.js`（无画面、无重力、随机数可复现） |
| `eval/eval.mjs`、`eval/decide.py` | 无界面地整局整局地玩，统计消行和存活块数；`--who model` 用 SDK 在本机 CPU 上跑 xDecision |
| `eval/table.mjs` | 把模型可能收到的 77 个输入全部问一遍，和代码判断的答案核对 |
| `../test/tetris_planner.test.mjs` | 检查每个计划落点经游戏函数执行后，棋盘和 planner 算的完全一样，以及重力暂停、局面和问题的格式等 |

```bash
node --test test/tetris_planner.test.mjs
node tetris/eval/eval.mjs --who rules            # 或 --who planner / --who random
node tetris/eval/eval.mjs --who model --model path/to/xDecision-Q8_0.gguf
node tetris/eval/table.mjs path/to/xDecision-Q8_0.gguf
```

## 许可

`vendor/tetris.js` 来自 vanilla-js-tetris（Maryele Trzaskos Gruber），它的 README 声明使用 MIT 许可，见仓库根目录的 `NOTICE`。
