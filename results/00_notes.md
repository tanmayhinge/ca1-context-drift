# The experiment behind DANDI 000718, in plain language

Source: Zaki, Pennington et al., "Offline ensemble co-reactivation links memories across
days", Nature 2024, doi 10.1038/s41586-024-08168-4, plus the session descriptions stored
inside the NWB files themselves (see `results/00_inspect.md`).

## The question

A mouse has a harmless experience on one day and a frightening one two days later, in a
different place. Does the fear leak backwards onto the earlier, harmless memory? The paper
says yes, and asks what the hippocampus does in between to make that happen.

## Terms used below

- **CA1**: an output layer of the hippocampus, the brain region needed for memory of places
  and events.
- **Miniscope**: a miniature microscope glued to the skull, which films a fluorescent protein
  that brightens when a neuron becomes active. One recording follows several hundred cells.
- **Ensemble**: the particular group of cells that was active during one experience.
- **Freezing**: complete immobility apart from breathing. In rodents this is the standard
  readout of fear. More freezing means the animal expects something bad.
- **Offline period**: time when the animal is resting in its homecage, not doing the task.
  Memories are thought to be replayed and stabilised then.
- **Representational drift**: the same place, revisited days later, is represented by a
  partly different set of cells. The code changes even though the world does not.

## The design

Three physical boxes, distinguished by floor texture, smell, lighting, fan noise and inserts:

| Context | What happens there |
|---|---|
| Neutral | 10 minutes of free exploration, nothing bad happens |
| Aversive (shock) | three brief foot shocks, 2 s each, one minute apart, after a 2 minute quiet baseline |
| Novel | never visited before the test, used as a control for generalised fear |

Day 1 neutral context. Day 3 aversive context. Days 4 to 6, one 5 minute test per day in the
aversive, the novel and the neutral context. Calcium imaging runs throughout, including
10 minute homecage recordings in between.

## What the paper found

- Mice froze more in the neutral context than in the novel context when the neutral visit had
  happened within a few days before the shock. Fear spread backwards onto a memory that was
  already formed. It did not spread forwards to a neutral context visited after the shock.
- Shock strength mattered. Only the strongly shocked group (1.5 mA) showed this linking. The
  weakly shocked group (0.25 mA) did not.
- During rest after fear conditioning, the cells that had encoded the neutral context were
  reactivated together with the cells that encoded the shock, more than chance allows, and
  more in the strongly shocked mice. This co-reactivation happened during waking rest rather
  than during sleep.

## The two mice in this dandiset, and why they are not replicates

The archive holds 2 of the 7 to 8 imaged mice. The shock amplitudes recorded in the files
show they come from opposite experimental groups:

| Mouse | Shock | Group in the paper | Sessions available here |
|---|---|---|---|
| Ca-EEG2-1 | 1.5 mA | strongly shocked, memory linking expected | neutral, conditioning, recall 1 |
| Ca-EEG3-4 | 0.25 mA | weakly shocked, no linking expected | neutral, conditioning, recalls 1 to 3 |

This is worth stating plainly in the report. Two animals from two different conditions are
not two samples of one population, so nothing here should be averaged across them.
