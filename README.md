# 媒体与认知大作业

Pascal VOC 2012 Object Classification

## 开发指南
为了方便代码管理，我们禁止直接 push 到 dev, master 分支上去，采用 rebase+merge request 的方式更新代码。大致操作如下

初始化分支
```bash
$ git checkout -b jzy # 这里创建自己的分支并 checkout 进去
```

写代码之前，先更新一下自己的分支：

```bash
$ git checkout dev
$ git pull # 更新 dev 分支
$ git checkout jzy
$ git rebase dev
```

完成代码编写，准备更新时：

```bash
$ git status # git add 前最好检查一下
$ git add .
$ git commit -m "Your comment"
$ git push origin jzy
$ git checkout dev
$ git pull # 更新 dev 分支
$ git checkout jzy
$ git rebase dev
```

注意，上面的 jzy 都换成自己的分支名字

然后进入github网页，提交一个 merge request ， dev 分支的拥有者 review 通过之后就会合并。
