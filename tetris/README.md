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

每来一个新方块，`ai.js` 做四件事：

1. **读局面。** 通过 `bridge.js` 拿到游戏自己的 `board`（10×20）、当前方块 `player.matrix` / `player.pos`，
   以及预览里的下一个方块 `player.next`。
2. **列出所有落点。** `planner.js` 按游戏的操作顺序（先旋转若干次，再左右移动，最后直接落下）找出每一个
   能到达的落点，算出落下之后的结果：消几行、新增几个洞、落点高低、表面平不平、边缘参差多少。旋转用的就是游戏自己的
   `rotate()`，碰撞检测和旋转时的踢墙逐行照抄 `tetris.js`，所以算出来的结果和游戏实际一致。然后按
   Dellacherie 特征（权重用 El-Tetris 公开的数值）加上对下一个方块的一步预判，给落点排序，取前 K 个（默认 4 个）作为候选。
   描述完全相同的候选在模型看来是同一个选项，只保留排在前面的那个。
3. **问 xDecision。** 把棋盘和候选交给模型，问一道单选题，候选按在棋盘上从左到右的顺序排，标成 A、B、C、D：

   ```json
   {
     "state": {"game": "Tetris", "board": "....##....\n#.######.#", "piece": "T", "next_piece": "L", "...": "..."},
     "questions": {"move": {
       "type": "choice",
       "instructions": "Where should the current Tetris piece be dropped? Clear lines when possible, never leave holes under the stack, and keep the stack low and flat.",
       "criteria": {
         "A": "clears nothing, covers no empty cells, lands lowest, surface 2 bumpier than the flattest, 2 more ragged edges than the tidiest",
         "B": "clears 1 line, covers no empty cells, lands 1 row higher than the lowest option, flattest surface, tidiest edges",
         "C": "clears nothing, covers 1 empty cell (new holes, bad), lands lowest, surface 4 bumpier than the flattest, 6 more ragged edges than the tidiest",
         "D": "..."
       }
     }}
   }
   ```

   模型给每个选项一个概率，概率最高的那个就是这一步。题型名不是写死的，加载后从模型的 `surface()` 里找
   “由调用方给选项”（shape 为 `named`）的那一种。
4. **执行。** 用游戏自己的 `playerRotate()`、`playerMove()`、`playerHardDrop()` 把方块放过去。执行前按
   落点再找一遍路径，模型思考期间方块如果往下掉了一格，也还是落到模型选的位置。

页面上会列出每一步的候选、模型给每个候选的概率、模型选了哪个（★），以及规则估值的第一名（⚑），可以对照
着看。“只用规则”模式不调用模型，直接取规则第一名，用来对比。

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

<!-- RESULTS -->

## 文件

| | |
|---|---|
| `index.html` | 页面；保留了 `tetris.js` 要找的元素 id（`tetris`、`next`、`score`、`lines`、`level`、`start-button`） |
| `vendor/tetris.js` | 上游游戏，原样 |
| `bridge.js` | 游戏内部接口 → `window.TetrisGame` |
| `planner.js` | 落点枚举、局面特征、给模型的题目（浏览器和 Node 都能用） |
| `ai.js` | SDK 启动、模型加载、每一步的决策和执行、界面 |
| `../test/tetris_planner.test.mjs` | 在 Node 里跑原版 `tetris.js`，检查每个计划落点经游戏函数执行后，棋盘和 planner 算的完全一样 |

```bash
node --test test/tetris_planner.test.mjs
```

## 许可

`vendor/tetris.js` 来自 vanilla-js-tetris（Maryele Trzaskos Gruber），它的 README 声明使用 MIT 许可，见仓库根目录的 `NOTICE`。
