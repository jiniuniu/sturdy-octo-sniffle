# Consumer Research Framework — Skill 开发指南

本文档帮助 agent 理解如何为 Consumer Research Framework 编写 skill。

---

## 第一步：读取服务引导文档

```
GET {base_url}/agent-guide
```

返回纯文本，包含：服务说明、关键业务概念、常用场景组合、最简调用流程。

## 第二步：读取完整接口文档

```
GET {base_url}/openapi.json
```

返回 OpenAPI 规范 JSON，包含所有接口的完整参数定义、类型约束和响应结构。

---

## 说明

skill 的具体实现（调用哪些接口、传什么参数、处理什么返回值）完全可以从上面两个接口推导出来，
不依赖任何特定的 agent 框架或目录结构。
