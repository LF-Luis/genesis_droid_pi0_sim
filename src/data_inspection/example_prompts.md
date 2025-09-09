see https://robo-arena.github.io/leaderboard under "pi0_fast_droid"

### https://autolab.berkeley.edu/assets/publications/media/2024-RSS-DROID.pdf
- "Close Waffle Maker"
- "Place Chips on Plate"
- "Clean up Desk"
- "Cook Lentils"
### https://www.physicalintelligence.company/download/pi0.pdf (from base model eval)
- Shirt folding: the robot must fold a t-shirt, which starts flattened.
- Bussing easy: the robot must clean a table, putting trash in the trash bin and dishes into the dish bin. The score indicates the number of objects that were placed in the correct receptacle.
- Bussing hard: a harder version of the bussing task, with more objects and more challenging configurations, such as utensils intentionally placed on top of trash objects, objects obstructing each other, and some objects that are not in the pre-training dataset.
- Grocery bagging: the robot must bag all grocery items, such as potato chips, marshmallows, and cat food.
- Toast out of toaster: the robot removes toast from a toaster.
- Fine-tuned eval
    - Franka items in drawer. This task requires opening a drawer, packing items into a drawer, and closing it. Because there is no similar task with the Franka robot in pre-training, we consider this “hard.”
### https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/
- "Place the yellow fish into the purple box"
- "Open the drawer"
- "Hand the pineapple to the programmer"
- "Pour water from the silver cup to the pink bowl"
- "Pick up all the objects into the basket"
- "Close the capsule lid of the coffee machine"
- It can grasp transparent objects
    - "Place the plastic bottle into the white cup."
    - "Place the plastic bottle into the bowl."
- It can grasp an object even when it is camouflaged into a colorful background
    - "Place the fish into the red box"
    - "Place the fish into the purple box"
- "Pick the pineapple and place it into the basket"
- "Remove the pink bowl from the tray"
- "Stack the wooden blocks"
- "Fold the cloth from left to right"
### Pick → Place (single object, single target)
- pick up the fork
- pick up the red block
- pick up the mug
- pick up the bottle
- pick up the pen
- pick up the blue cup
- pick up the cloth
- pick up the lid
- pick up the green block and place it on the mat
- put the bottle in the bowl
- put the mug in the bowl
- put the pen inside the cup (from visualizer)
- take the bottle from the table and put it in the bowl (from visualizer)
- put the blue cup in the white bowl
- place the red block in the basket
- place the cup on the coaster
- place the mug on the plate
- place the block on the square target
- place the bottle next to the bowl
- put the marker in the pencil cup
### On/On-top/Stack (table-top)
- put the orange block on the green block (from visualizer)
- stack the two green blocks
- place the cup on top of the upside-down cup
- put the lid on top of the pot
- place the mug on top of the coaster
- stack the red block on the blue block
### Into/Inside/Containerizing
- put the spoon in the bowl
- put the fork in the bowl
- put the cup in the drawer (drawer open)
- put the pen inside the drawer (drawer open)
- place the marker in the cup
- put the bottle in the bin
- place the block into the box
### Move/Slide/Nudge (no grasp or gentle re-pose)
- slide the blue block to the right edge of the mat
- move the mug to the right side of the table
- push the bottle closer to the bowl
- nudge the red block away from the edge
- move the cup to the left of the green block
### Open / Close (single step)
- open the top drawer
- open the bottom drawer of the file cabinet to the right (from visualizer)
- close the drawer
- remove the lid from the pot
- put the lid back on the pot
- open the box
- close the box
### Short two-step combos (difficult, may need to do step-by-step)
- open the top drawer and put the pen inside, then close it (pattern seen in DROID eval “Clean up Desk”)
- remove the lid and put it on the table
- pick up the mug and place it in the drawer (drawer open), then close the drawer
- pick up the bottle and put it into the bin, then push the bin back
### Place with simple spatial qualifiers
- put the block to the left of the bowl
- place the cup in front of the plate
- put the marker behind the mug
- place the bottle between the cup and the bowl
### Attribute references (safe & common)
- pick up the red block
- pick up the green cup
- put the blue lid on the pot
- place the tall bottle next to the small bottle
### “Turn” / “Press”
- turn the knob clockwise (appears in DROID “Cook Lentils” eval)
- press the red button
- flip the switch up
### More examples (mix of the above)
- put the cloth in the bowl
- place the mug next to the plate
- put the cup into the drawer (drawer open)
- place the block on the target sticker
- pick up the pen and place it in the cup
- move the bottle to the back-right corner of the mat
- slide the cup toward the camera
- nudge the block away from the cup
- put the lid on the container
- remove the lid and set it down on the mat
- open the drawer and place the marker inside
- close the drawer
- pick up the fork and put it in the bowl
- place the green block near the red block
- move the mug closer to the bowl
- put the bottle in the basket
- place the cup on the square
- put the pen on the notebook
- place the block inside the box
- push the bottle to the left
- place the mug in front of the bowl
- pick up the cloth and put it in the bin
