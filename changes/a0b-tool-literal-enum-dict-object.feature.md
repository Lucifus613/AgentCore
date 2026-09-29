`Literal[...]` tool parameters are published as a typed `enum` (a nullable `Literal` also admits `null`), and a bare `dict` / `dict | None` parameter as `{"type": "object"}` instead of `string`.
