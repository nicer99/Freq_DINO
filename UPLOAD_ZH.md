# 仓库范围与发布说明

仓库地址：[nicer99/Freq_DINO](https://github.com/nicer99/Freq_DINO)。

本仓库仅发布方法实现、训练配置、数据准备工具和必要框架。模型权重、数据、训练日志、实验输出、论文文件及原项目 Git 历史均不属于发布内容。预训练模型由使用者按安装文档从原提供方自行获取。

## 本地使用

```bash
git clone https://github.com/nicer99/Freq_DINO.git
cd Freq_DINO
```

先阅读 [README](README.md) 和 [安装说明](docs/INSTALL.md)。安装依赖后可检查：

```bash
python scripts/run_experiment.py train --method freq_prior --dry-run
python scripts/check_method.py
```

第一条只显示训练命令；第二条使用合成张量在 CPU 上检查方法前向和梯度，均不读取研究数据或权重。

## 后续更新

`.gitignore` 排除了常见权重文件、`data/`、`weights/`、`work_dirs/` 等本地产物。提交前仍应检查 `git status --short` 和 `git diff --cached --stat`，只提交源码及文档。

原创代码暂不指定许可证。保留的 Apache-2.0 文本适用于上游 MMDetection；具体范围见 [LICENSE.md](LICENSE.md)。
