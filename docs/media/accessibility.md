# Accessibility Guidelines

Ensure all multimedia artefacts meet accessibility standards alongside the
written documentation overhaul.

## Captions and Transcripts

- Generate captions for every video and verify timing alignment manually.
- Publish transcripts in Markdown and reference them from the parent tutorial or
  playbook using `[Transcript](transcripts/<file>.md)` links.

## Visual Design

- Adhere to WCAG 2.1 AA contrast ratios when designing slides or overlays.
- Provide dark-mode friendly palettes and avoid colour-only encodings. Include
  texture or icon cues to differentiate series when embedding charts.

## Navigation

- Ensure embedded players are keyboard navigable; prefer iframe embeds that
  expose controls to assistive technologies.
- Supply descriptive `title` attributes for all multimedia elements.

## Verification Checklist

- [ ] Captions reviewed by a human editor.
- [ ] Transcript linked from the relevant tutorial or playbook.
- [ ] Colour contrast validated with automated tooling (e.g. `pa11y`).
- [ ] Keyboard navigation confirmed on desktop and mobile browsers.
