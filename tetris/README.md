# xDecision 玩俄罗斯方块

让决策模型 **xDecision** 在浏览器里自动玩 [vanilla-js-tetris](https://github.com/trzaskos/vanilla-js-tetris)。
模型通过 webtorch SDK 在本机显卡上运行，每来一个方块，就由模型决定把它放在哪里。操作游戏时直接调用
游戏自己的函数，不模拟键盘，也不读屏幕像素。

```bash
node serve-coi.mjs . 8119
# 打开 http://localhost:8119/tetris/
```

部署到 Pages 后的地址是 `https://xnetsc.github.io/webpytorch/tetris/`。

## 用的是哪个模型

就是 Pages 聊天应用里列出的那个 xDecision（`chat/models.json`）：

| | |
|---|---|
| 仓库 | `mccoysc/xDecision` |
| 文件 | `models/gguf/xDecision-Q8_0.gguf`（402,546,752 字节） |
| 来源 | 默认从 ModelScope 读取（`webtorch.modelscope_read()`，.cn 和 .ai 两个站点都会试），也可以选 Hugging Face |

加载和提问的写法和聊天应用一样：

```js
const wt = await webtorch.start({ baseURL: '../' });
await wt.run(`import webtorch
webtorch.set_io_read(webtorch.modelscope_read())
webtorch.set_io_write(webtorch.default_io_write)`);
const info = await wt.load('mccoysc/xDecision', { file: 'models/gguf/xDecision-Q8_0.gguf' });
const res  = await wt.decide(state, { move: question });   // res.answers.move.choice
```

模型下载一次之后保存在浏览器缓存里，下次打开不用重新下载。手里已经有这个文件的话，也可以从自己的服务器
加载：`/tetris/?model=/models/gguf/xDecision-Q8_0.gguf`（同源，或者对方允许跨域）。

## 每一步怎么决定

xDecision 根据给定的事实回答问题，不会推理。这里把它当一个便宜的 if-else：代码把事实写成标签，规则写成题目，
模型挑出标签符合规则的那个选项。

每来一个新方块，`ai.js` 做四件事：

1. **读局面。** 通过 `bridge.js` 拿到游戏自己的 `board`（10×20）、当前方块 `player.matrix` / `player.pos`，
   以及预览里的下一个方块 `player.next`。
2. **列出所有落点。** `planner.js` 按游戏的操作顺序（先旋转若干次，再左右移动，最后直接落下）找出每一个
   能到达的落点。旋转用的就是游戏自己的 `rotate()`，碰撞检测和旋转时的踢墙逐行照抄 `tetris.js`，所以算出来的
   结果和游戏实际一致。然后按 Dellacherie 特征（权重用 El-Tetris 公开的数值）加上对下一个方块的一步预判排序。
3. **把事实写成标签，问 xDecision。** 每个落点只有两条事实：消不消行，留不留新洞。从前 K 名（默认 4 名）里，
   每一类标签只留排在最前的那个，所以选项最多 4 个，而且标签各不相同。题目就是规则本身：

   ```json
   {
     "state": {"game": "Tetris", "piece": "T"},
     "questions": {"move": {
       "type": "choice",
       "instructions": "Choose the option that clears a line and is hole-free.",
       "criteria": {
         "A": "hole-free",
         "B": "clears a line, hole-free",
         "C": "makes a new hole"
       }
     }}
   }
   ```

   这道题相当于：有既消行又不留洞的就选它；没有，就选不留洞的；再没有，就选能消行的；都不行，就是剩下那个。
   模型选概率最高的选项。题型名不是写死的，加载后从模型的 `surface()` 里找“由调用方给选项”（shape 为 `named`）
   的那一种。

   标签只有 4 种，所以模型可能被问到的输入一共只有 11 种标签组合 × 7 种方块 = 77 个。选项按固定的标签顺序排列，
   这 77 个输入都实际问过模型，全部答对（`node tetris/eval/table.mjs <模型文件>` 可以重新检查）。按落点在棋盘上
   从左到右排时，在所有排列下只答对 364 / 420 个，所以顺序是固定的。在这个顺序里，正确答案有时排第一、有时第二、
   有时第三，说明模型读的是标签，不是位置。
4. **执行。** 用游戏自己的 `playerRotate()`、`playerMove()`、`playerHardDrop()` 把方块放过去。执行前按
   落点再找一遍路径，模型思考期间方块如果往下掉了一格，也还是落到模型选的位置。

页面上会列出每一步的选项、模型给每个选项的概率、模型选了哪个（★），以及同一条规则在代码里判断出的答案（✓），
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

| 谁来选 | 平均消行 | 平均放下方块 | 500 块内输掉 | 和规则答案一致 |
|---|---:|---:|---:|---:|
| **xDecision**（页面默认） | 198.6 | 500 | 0 / 5 | 100% |
| 同一条规则，在代码里判断（“代码”模式） | 198.6 | 500 | 0 / 5 | 100% |
| 规划器排第一的，不用规则 | 198.4 | 500 | 0 / 5 | 86.7% |
| 在同样的选项里随机选 | 19.4 | 85 | 5 / 5 | 56.3% |

在没有显卡的机器上，用 SDK 的 CPU 路径（numpy）跑，每步约 100 毫秒。浏览器里走 WebGPU，权重相同，但数值
路径不同，所以 77 个输入的检查是针对 CPU 路径做的。

### 这个模型能做什么、不能做什么

在 240 个真实对局局面上实测（每个局面 4 个候选），结果决定了上面的设计：

| 问法 | 结果 |
|---|---|
| 选项里写着题目要的事实，问“哪个选项是 X” | 99%～100% 答对 |
| 没有选项符合时，答“都不是”（`none`） | 3% 答对：几乎从不选 `none` |
| 事实放在 state 里，按字母问“选项 B 不留洞”（是/否题） | 所有问题都答“是” |
| 把一句事实分到几个只差一个“不”字的类别里 | 51% 答对 |
| 每个选项写一串相对比较（“落点最低、最平、边缘最整齐…”），让它挑最好的 | 63% 选中规划器第一名，整局平均 106 行、300 块就输 |
| 选项写成数字（`stack 8 high, bumpiness 10`） | 240 次里 218 次选最后一个选项，跟内容无关 |

所以：选项只写正面的事实标签；不设“都不是”；不让它按字母去 state 里查；标签不重复；规则用标签自己的原话写成
题目。这样它就是一个便宜、可靠的 if-else。

### 浏览器

完整流程在无界面 Chromium 里跑过：通过 SDK 加载 xDecision、模型连续决策并落子、候选标在棋盘上、“代码”模式、
游戏结束后自动重开、停止后重力恢复、手机宽度没有横向滚动。那台机器没有显卡，WebGPU 是 CPU 模拟的
（`swiftshader`，`isFallbackAdapter: true`），每步约 8 秒（模型每步只读 46 个 token 左右）；在真正的显卡上会快得多，但没有在显卡上实测过这个页面。

## 文件

| | |
|---|---|
| `index.html` | 页面；保留了 `tetris.js` 要找的元素 id（`tetris`、`next`、`score`、`lines`、`level`、`start-button`） |
| `vendor/tetris.js` | 上游游戏，原样 |
| `bridge.js` | 游戏内部接口 → `window.TetrisGame` |
| `planner.js` | 落点枚举、局面特征、事实标签和规则题目（浏览器和 Node 都能用） |
| `ai.js` | SDK 启动、模型加载、每一步的决策和执行、界面 |
| `eval/headless.mjs` | 在 Node 的 vm 里跑原版 `tetris.js` + `bridge.js`（无画面、无重力、随机数可复现） |
| `eval/eval.mjs`、`eval/decide.py` | 无界面地整局整局地玩，统计消行和存活块数；`--who model` 用 SDK 在本机 CPU 上跑 xDecision |
| `eval/table.mjs` | 把模型可能被问到的 77 个输入全部问一遍，和代码里的规则核对 |
| `../test/tetris_planner.test.mjs` | 检查每个计划落点经游戏函数执行后，棋盘和 planner 算的完全一样，以及重力暂停、题目格式等 |

```bash
node --test test/tetris_planner.test.mjs
node tetris/eval/eval.mjs --who rules            # 或 --who planner / --who random
node tetris/eval/eval.mjs --who model --model path/to/xDecision-Q8_0.gguf
node tetris/eval/table.mjs path/to/xDecision-Q8_0.gguf
```

## 许可

`vendor/tetris.js` 来自 vanilla-js-tetris（Maryele Trzaskos Gruber），它的 README 声明使用 MIT 许可，见仓库根目录的 `NOTICE`。
