# Panda (no-hand) + Robotiq 2F-85 — MJCF

This file combines two open-source MJCF assets into one model.

## Attribution
- **Franka Emika Panda (no-hand)** — Apache-2.0
  Source: https://github.com/Genesis-Embodied-AI/Genesis/tree/main/genesis/assets/xml/franka_emika_panda
  License text: [LICENSE-panda_no_hand](./LICENSE.Apache-2.0.Genesis-Panda)

- **Robotiq 2F-85 gripper** — BSD-2-Clause
  Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85_v4
  License text: [LICENSE-robotiq_2f85](./LICENSE.BSD-2-Clause.Robotiq-2F85)

## What I changed
- Merged the two assets into a single MJCF and wired up the gripper to the arm.
- Minor integration edits (names/includes/etc).
