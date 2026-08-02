---
name: event-collection
description: Use this skill when the user provides travel details like origin, destination, dates, purpose, or when planning a trip. It extracts structured event information for itinerary planning.
---

# Event Collection Skill

这是一个内部辅助技能，用于从用户输入中提取结构化的行程信息，包括：

- 出发地、目的地、起止日期和行程天数；
- 去程与返程的大概时段；
- 去程、返程和住宿是否已经预订；
- 用户明确提供的已预订交通与住宿详情。

预订状态只允许使用 `confirmed`、`reference` 或 `null`：

- `confirmed`：用户明确说明已经预订，并提供的详情可作为确认事实；
- `reference`：用户未预订、不需要推荐或选择先看参考方案；
- `null`：用户没有说明，必须交给协调器继续询问。

酒店和航空偏好不等于已经预订，不能据此生成 `confirmed`。

通常由 `IntentionAgent` 自动调度，配合 `plan-trip` 技能使用。
