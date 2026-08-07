# WiFi terminal (barista-wifi branch)

Goal: the board broadcasts its own WiFi network. Anyone joins from a phone,
gets a retro terminal page, types a question, and watches an LLM answer stream
back word by word. No internet involved at any point.

## Target model

slvDev's barista (espresso Q&A, 8-9M params, ~18 pieces/s on this board). It
already takes typed questions over serial, so the access point replaces the
serial line as the input. The interactive loop exists; only the transport is
new. Barista's runtime lives in the upstream esp32-ai repo; this branch either
vendors the needed pieces or documents building against a sibling checkout.

## Architecture, decided

- SoftAP, open network, memorable SSID. DNS catch-all so joining pops the page
  like a captive portal.
- One HTML page served from flash: green phosphor terminal, scanlines,
  blinking cursor, live tokens-per-second readout.
- Answers stream over Server-Sent Events so words appear as the model produces
  them. SSE over a raw socket is trivial to hand-roll and needs no websocket
  library.
- Generation must not block the HTTP accept loop: model runs in its own task
  (the record42 firmware already demonstrates the pattern), tokens go into a
  ring buffer the SSE writer drains.
- Single client is fine for v1. Queue or reject a second asker politely.

## Open questions for when work resumes

- WiFi + PSRAM power draw together: watch for brownout on weak USB ports.
- App partition budget once the WiFi stack links in (barista's app was 461KB
  in a 1MB partition, so there is room, but measure).
- Whether to show the OLED mirroring the web answer, or stats only.

## Status

Not started. Everything above is design settled during the record42 build.
The record firmware's task layout, serial buffering fixes, and display code
carry over directly.
