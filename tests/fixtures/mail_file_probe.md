# Mail File Probe

This fixture is used by the real agent mail attachment test.

The attachment-only secret is:

MAIL_FILE_PROBE_GREEN_LANTERN_20260429

Chinese text for UTF-8 verification:

雾港码头的蓝色台灯在午夜亮起，收信人必须按普通文本读取这一行。

Expected behavior:

- Sender agent attaches this file to an AgentMail message.
- Receiver agent reads the routed attachment as UTF-8 text.
- Receiver agent replies with the attachment-only secret and the blue-lamp sentence.
