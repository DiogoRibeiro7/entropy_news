# Storyboard Template

Use this storyboard when planning video walkthroughs for Entropy News releases
or tutorials. Each panel should capture the key message, on-screen elements, and
supporting narration.

| Panel | Objective | Visuals | Narration |
| ----- | --------- | ------- | --------- |
| 1 | Introduce the scenario (training, forecasting, or rollout) | Title card with release version and owner | "Welcome to the Entropy News Release X.Y walk-through" |
| 2 | Demonstrate the workflow | Screen capture of CLI or dashboard | Highlight the command sequence or dashboard controls |
| 3 | Surface insights | Charts, metrics, or causal plots | Explain what the numbers mean for stakeholders |
| 4 | Next steps | Links to tutorials, playbooks, and API docs | Summarise follow-up actions and support channels |

## Tips

- Keep each video under five minutes and include captions generated with tools
  like `ffmpeg` or `YouTube Studio`.
- Store raw footage and final renders alongside the relevant registry entry (see
  :doc:`../playbooks/research_registry`).
- Embed videos into documentation pages using HTML blocks:

```{eval-rst}
.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/VIDEO_ID" title="Entropy News Overview" frameborder="0" allowfullscreen></iframe>
```

- Provide downloadable transcripts in Markdown and link to them using standard
  MyST syntax for accessibility compliance.
