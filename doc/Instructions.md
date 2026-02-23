**HAR Dataset Labeling workflow**

**Introduction**  
The aim of this project is to complete the egocentric Ego4d video dataset, to make it relevant for activity recognition. What we did here is to use already existing written narration annotation to add high level scenario and low level action annotations. We used LLMs and semantic validation for a total of 355k rows. Your mission here, shall you accept it, would be to check a part of these labels, to make sure that this dataset is reliable. We want to achieve a total of 20% of labeled checked. If the accuracy is good enough, we will be able to publish the dataset.   
There will be 3 different rounds: between each round we will be fine tuning our LLM to improve the remaining labels. We will precise the deadlines soon.

**I and II only applies to people who have vpn access**

**Lets get to it\!** 

1. **Connect**  
1) Connect to office wifi or VPN (send us a DM if it’s your first time on the VPN)  
2) Go on [Label-Studio](http://10.100.241.227:8080/user/signup/?token=qdLXG6rPWAszso4iho41vS6lLFn5vQa4E3qfE8ml), create an account, and go to **HAR\_dataset\_Deduped**

**![][image1]**

**II.     Filter Your Batch (Keep in mind, always have the filter on)**

1) Look up your **batch number** in the assignment table below  
2) In the dataset view, click **”Filter”** → add a filter: **`batch`** \= your batch number (e.g. `1`)  
3) Add another filter: **`round`** \= `r001`  
4) Now you should only see the tasks assigned to you

 **III.     Labeling**

1) \[Optional\] Click on “column” and select these to make it more clear

**![][image2]**

2) With your batch filter active, click on the first task to start labeling. Read the instructions carefully on your first task to understand what is good and what is not.  
3) Label through your batch(do not filter to other’s batch) — you can stop and resume anytime, your filters will stay.  
4) To check your progress, go back to the dataset view and add a filter: **”annotated\_by”** → select your name. The task count shows how many you have completed out of your 2,500.(not necessary needed, enter batch number would work as well)

	

**III.  How many rows should you label**

For the 1st round, we will ask you to label **2500 rows**. If you take max 5sec per label, it will take between a but less than 3h (we are 11 participants for now). We will ask you to complete each round in a maximum of 3 days to meet the deadlines.

You can put your name in the tab, and label your corresponding batch tasks. Just filter the rows in Label Studio with your batch number. 

Please come back and check, this number might go down with the number of persons helping\!\!

| Batch Number | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Name | Léo | Kida | Faizaan | Julia | Serena | Andrew | Todd |  | Maher | Sinclair | Harry | Arnav |

**Thank you again for your help\!\!\!**

**HAR DATASET LABELING Logic — ROUND 1**

**WHAT YOU’RE DOING**

*   You are checking LLM-generated action labels for egocentric videos.
*   Read the **narration** (the primary signal) and the **scenario** (optional context), then see what the LLM classified and decide if it is correct.

**IMPORTANT: About Narrations and Scenarios**

*   The **narration** describes a single action and is **always your primary basis** for labeling.
*   The **scenario** describes the overall activity of the video (e.g., “Cooking”, “Walking Outdoors”). Videos can be 20+ minutes long and contain many different actions within a single scenario.
*   **Narration and scenario may seem unrelated** — this is normal. For example, during a “Walking Outdoors” scenario, the person may stop to take liquid from a bottle. The narration describes what is happening at that specific moment, not the overall scenario. **Always label based on the narration, not the scenario.**
*   The scenario is provided as optional context — it can sometimes help you decide between two classes, but it should never override the narration.

---

**THE 5 ACTION CLASSES**

These 5 classes are designed to be **mutually exclusive** — each narration should belong to exactly one class. We know that some actions feel like they could fit two classes (especially Essential Operation vs Object Transfer). The decision rules below tell you how to resolve this.

1.   **Object Transfer**: An object changes location — picking up, putting down, handing, opening/closing, dropping
2.   **Essential Operation**: A purposeful task action that transforms, operates, or works on something — cutting, washing, typing, screwing, stirring, sewing
3.   **Stationary**: The person’s body is still or only a limb moves — idle, waiting, talking, resting, holding
4.   **Locomotion**: The person’s whole body moves through space — walking, running, climbing, standing up, sitting down
5.   **Search**: The person is visually scanning or inspecting — looking around, checking, searching for something

**The Critical Rule: Essential Operation vs Object Transfer**

Many actions involve moving an object AND doing a task. Use this rule:

> **Ask: “Is the ACTION itself the point, or is MOVING THE OBJECT the point?”**

*   If the verb describes **operating/transforming** something → **Essential Operation**
    *   “cuts the onion” → EO (cutting is the task)
    *   “stirs the soup” → EO (stirring is the task)
    *   “types on the keyboard” → EO (typing is the task)
    *   “washes the dishes” → EO (washing is the task)
    *   “screws the bolt” → EO (screwing is the task)

*   If the verb describes **displacing** an object from A to B → **Object Transfer**
    *   “puts the pan on the stove” → OT (moving the pan is the point)
    *   “picks up the knife” → OT (picking up is displacement)
    *   “opens the fridge” → OT (opening = displacing the door)
    *   “drops the glass” → OT (displacement, even if unintentional)

**Quick test:** If you can replace the verb with “moves [object] from here to there” and the meaning is preserved → **Object Transfer**. If not → **Essential Operation**.

---

**HOW TO LABEL**

  Gold (hotkey 1\)  →  The LLM label is correct

  Bad  (hotkey 2\)  →  The LLM label is wrong → select the correct class

  Skip (hotkey 3\)  →  You genuinely can’t tell (use sparingly, \<5%)

  Delete (hotkey 4\)  → Something is really wrong with the line (Only when the reasoning and annotation is messed up, or the true case doesn’t belong to any of these classes)

**When an action could plausibly be TWO classes:**

If the LLM label is reasonable but you think another class is equally valid:
1. Mark it **Gold** (the LLM’s choice is acceptable)
2. **Also select the alternative class** as an additional option (use the checkboxes)
3. We will review these dual-labeled cases during data analysis

This is better than marking it Bad (which implies the LLM is clearly wrong) or Skip (which loses information). Reserve Bad for cases where the LLM label is **clearly incorrect**.

---

**COMMON CONFUSIONS**

| Narration | Correct Class | Why |
| :---- | :---- | :---- |
| “C moves the pan” | Object Transfer | Object is being displaced; person is not moving through space |
| “C puts the pan on the stove” | Object Transfer | Moving an object from A to B |
| “C stirs the soup” | Essential Operation | Stirring is a task action, not displacement |
| “C cuts potatoes” | Essential Operation | Cutting transforms something |
| “C opens the fridge” | Object Transfer | Opening = displacing the door |
| “C checks the time” | Search | Visual checking = search |
| “C drops the bowl” | Object Transfer | Unintentional displacement is still transfer |
| “C stands up” | Locomotion | Whole body moving through space |
| “C sits down” | Locomotion | Whole body changing position |
| “C adjusts the camera” | Essential Operation | Operating a device |
| “C looks around” | Search | Visual scanning |
| “C looks at the booster” | Search | Visual inspecting |
| “C looks at person A” | Skip | Hard to determine intention |
| “C holds the phone” | Stationary | Holding without moving = still |
| “C talks” | Stationary | No body movement |
| “C bends down” | Locomotion | Body moving through space (if changing location) |
| “Closes \#unsure with both hands” | Object Transfer | Closing = displacing an object |

---

### **Further Explanations:**

**Buttons/Validation Choice:**

* **Gold:** The LLM label is correct or at least reasonable. If you think another class is also valid, mark Gold AND select the alternative.
* **Bad:** The LLM label is **clearly wrong** — there is an obviously better class. Select the correct class.
* **Skip:** You genuinely cannot decide. Use sparingly (\<5%).
* **Delete Row:** The row itself is broken — missing data, corrupted text, or clearly unusable (e.g., “\#Summary” rows, gibberish text).

**5 Action type guidelines:**

**Object Transfer:** An object is **displaced** from one location to another. The focus is on the movement of the object, not on performing a task.

* Examples: “picks up the knife”, “puts the book on the table”, “drops the glass”, “opens the fridge”, “puts the pan on the stove”.

**Essential Operation:** A **purposeful task action** that operates on, transforms, or works with something. The focus is on the skill/task being performed, not on moving an object.

* Examples: “cuts potatoes”, “stirs the soup”, “types on the keyboard”, “washes the dishes”, “paints the wall”.

**Search:** The person is **visually scanning or inspecting** — actively looking around or checking something.

* Examples: “looks around”, “searches the drawer”, “checks the time”, “scans the room”.

**Locomotion:** The **whole body moves** through space, changing the person’s position in the environment.

* Examples: “walks down the stairs”, “walks across the room”, “stands up”, “sits down”, “climbs the ladder”.

**Stationary:** The person’s body is **mostly still** — only limbs or small parts move, or the person is idle.

* Examples: “sitting”, “holds the phone”, “waves hand”, “talks”, “reads”.

---

**FAQ (from annotator questions)**

**Q: “puts the pan on the stove” — is this Essential Operation or Object Transfer?**
A: Object Transfer. The action is moving the pan from one place to another. If it were “heats the pan on the stove” (cooking task), that would be Essential Operation.

**Q: A lot of Essential Operations also involve moving objects. Should I skip these?**
A: No — use the “what is the POINT of the action” test. “Cuts the onion” involves moving a knife, but the point is cutting (EO). “Picks up the knife” is about getting the knife from A to B (OT). Do not skip just because two classes seem possible — pick the best one, or use Gold + alternative.

**Q: The narration and scenario seem completely unrelated. Is this an error?**
A: No. Scenarios describe long videos (10-20 min). The narration describes one moment within that video. A “Walking Outdoors” scenario can include moments where the person stops to eat, check their phone, or pick something up. **Always label based on the narration.**

**Q: What does \#unsure mean in the narration?**
A: It means the original Ego4D annotator could not identify the object. Treat it as “some unknown object” and classify based on the verb. Example: “picks \#unsure” → Object Transfer (picking up = displacement).

**Q: What if the narration is too vague to classify (e.g., “C moves”)?**
A: If you genuinely cannot determine the class, use Skip. But try to use the verb as a guide first.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAb4AAACfCAYAAACGCrzOAAAS7klEQVR4Xu3d+Z/UxZ3H8fwlBiRL8Fg3JgpB2ZW46qJIRJRLVowShTWgonKMXHIjSEAOkcRE45EDgkcS12BcE42aBZP1enhEHmxMXCNzMsNczEz7+O58aqgv1Z/6ds+3p3tmuqnXD8/H9LeqvvX9fnvg++6qb83MF04f+sUIAIBQfEEXAABwKiP4AABBIfgAAEEh+AAAQSH4AABBIfgAAEEh+AAAQSH4AABBIfgAAEEh+AAAQSH4AABBIfgAAEEh+AAAQSH4AABBIfgAAEEh+AAAQSH4AABBIfgAAEEpSfBN//qQaOOEodHuawAA6B8PThoaLbhkaDTyDD+HClFU8H1n7JDohZuGRUcWfAkAgAHzo6lDoy8P83MpjT4H3+ZvDvVOBACAgfLszGHRqDP9fOpNn4JPRnr6BAAAGGiPTD3dy6jeFBx8MreqDwwAwGAp9JlfwcEnDxb1QQEAGCySSzqr8ik4+GRVjT4oAACDRXJJZ1U+BQefLCnVBwUAYLBILumsyofgAwBUNMmloUNO8/Iql4KCb+gQgg8AUF76OfhOI/gAAGWF4AMABMUG39CE3EpC8AEAKlocfEP83EpC8AEAKhrBBwAICsEHAAjKyeBLt8CF4AMAVLTyDr7FI6L2916MmvYt8+qanloRtb3za69cfP55xtDl9duvNf1ZbW/+Kqp/aIbXLq1j/7nJ9KPLK0nL7x+N34/Wg3uihodneW36wvapyweKHPvYr7d45QBQ1sFXfc/ZJsA6jxzy6o4fPpAYbg3fuyEOvmO/ui+rrumZlXFdlkyX108aHZ/9OfEckmTaj6VuWyw5ztHHb/PKk3S1NPjvh+z/xO1e20LYfnT5QJFjd9Yc9soB4JQLPgkx9wbu1tnga3ntibj/TEe7Ketq/MzrqzenSvBluo6b1zIi7qz/JH7v6ndN99qnlfT+DySCD0Aup17w2dA7EYAyXWrrdPDpfXRfWsP3vxVlMp2mbftHr3rB1/zCtijT2ROk7k23q6k6Poa8FrZO2tnzbX5xZ9bxZCrVXkf9jilZdTLNa/tsPfAz/1jd+8nr2nUXZe2nucFnNT27JvE9Of7x/8TXoPtp//CV+LgSmO7+0r71wJ64be2Gi02ZfD9svWj4/o1mf/kwUrNmTFb/tevHRpnjraZP+RBRvezcrPquxiM9dd1t5HuuvwcAysv9i/dHexdtiLcPLJySta3rS6kigi/T0RYd/dF/ZOlqrvNvzP/7hilr/s2OqOEHs3pufg2fxvVJwVe3bVLiTV6r2zIhbqdJfccn73rldgrVKz+xjy4TNvyanlru1UkwSJ0dpbrMDT+hz9pN47xrcSUFn9uP3nbFfRyr9ercNvK149P34/byXFXKWl75Yc6+zb4nPrTINeg6If8+8u1P8AHlSQJtXlWtYcvstgSgu633LYWKCL583Pa6TG/Hz/jMqKItq5/epgZtu6ZfrO05t+VfzepfgjHT1hS3lzBxj5001dny6mPdI7fnvWPI6866v5rXdVuvMtsy+pOvzS/tNuUdzig47vtEUKS5HitX8Nlnf3KdHf/3nnnd8vqTps4Gl5yj+z2y+7ofAmTbnG+K4Dt++KDZth9g7HnFfZ24vpp7R/bUd4++ZYGSfW37t6Nkgg8oTxJuOth62y6ligg+Gc3ICM5lp/RsW3szFY177zHstKNd5JJrcUvdAxO9Y2vxzTdPmdzc9TPGxp/ebeqSgk+mITOtjd75SJ2srnTL7Ogm097stbda39gXn1exweeeiz6OS0ao8rX9/Zfy7p8m+PLtn4v9AGPfH9P/9mtNGcEHIElFBF+aZ3w6cDRpo6c67TOj5v/a5fWvuf0klbnHsiEn7HMsHXzuSElGK+4+to0JCOe6qpecE09zyvM2TZ7N2XMpJvhqVo1OvDZ9PNHy8sOmrvWPT2f1ofeX6VBbJyNHKSs0+PSxhf1wc2Th8Hjf2vXfMGUEH4Akp07wnbg51qw4L0t8A108wgs+dz/dv2bb1W27xivTr93txr1VZlsHn/zMnGy3Htzr7SOvq6vOijpr/5JV1/7B78zUqLx2b+rH9m8x04NuWwkku52PDr6GR+fE59H29nM9bRo/O3GuJxeo2AUoNasv9K695ZVHssp0vd3WwSfXIdvNv/1eT5l6RlpddebJPmQRTffI7vih101dV0v9yXM78YGG4AOQ5JQIPrvSUlb+6XZHFn05vnEmBV/NylFxvbevYttZcl52v95+Hk6mU93ypP7cOgnMpHJx7Ln7vDp3Ec/xj9+My9ve6gmvXBLPO+HnGtsPvea1s3V6Wla3af3DT7LK7EpWCUip1/sY6hy8+s9PTm/a1bVW19G/x8fR1wEAZR18g0HfXIVbLysrm55d7e0nZNGF/JaZxAAW3aNOefZYu/GyuEymMxv3LPbbntCw+/qo8WeLvHIh+x19bK5XLo4+eYcZvclrfT1J15VK94eIpmdWZT1Pc8lv02l4ZLZXbskCHffaLfd8ZMWuXdCjyQIi+fAi77Ouq7n3fLMStrdVrABA8CmyMlPTbSqNvp5yu64+BzEA9AHBh0FH8AEYSAQfACAoBB8AICgEHwAgKAQfACAoBB8AICgEHwAgKAQfACAoBB8AICgEHwAgKAQfACAoBB8AICgEHwCgKM37t3pl5awigq92w8VR9dKveOWa/Lmagfhlx/U7p2b9UVQACNlA3HdLqayDT/4Wnf4bctVLzvHaWZ11f83553bkL3Tbb07H3972+hXtH75s6nV5xyfvmHL5G3e6Th/nyMLhpryz/m9+HYABkfP/ZwHkb0Pqskok19GfH9RltCfvdf2u67y6UivVtZR18Jkw+vPv420Jr0xnu9fObS+jsaRy+Yveuf4j1Ky+MKsuV7vqZeeavzbutmt7d39WG/sfjuADBof9P3j80OteXRqZjra4D5Hvw3a5c69D7oG6vhTcY+i6Uml794WSXktZB5/W/NLunG+u/AXuXHVWrnop7zxyqNd2mrz57R++Em+3v//bnr66R54EHzDw7I2xr6HX8trjWf//JfTyfdguZ5lMZ1S3ZUK8LdfV8en7Xru+kNFdVhAppXzmV7NqtOmzlNdSUcFn/kEfPuCVCzPN2d7slev9dZmMEHW5bMs/GvnaWXPY28f9BtsyeQYp2/KV4AMGnv0/2dfQy0XfHyqFPu/6ndO8sr6y05v52LbFHrOruc57hFXstVRE8NVuGmcuMnO81auzpF6eCepy3SaprP3Qa15Z075l0ZHFI8zrrsbPsurlmxBPnS4cfrKfD35nXhN8wMCyN9tSh17rn57J+WG7GHUPXJ2KLOzT+6Yho6Nc9ztdVgwdgEnP+XKVpyX7tx7ck1iuy9Iq++CTCzah8tGrXp1Vs+K8VG+CbtP01HKvTKvfMSVnm66jf48yHT3TINKm7e3nDBl5SkjL67pt13j7AShMvkDrr9CTkUau//vFcsMin8Y9i7190zj62NzEc08qK5a879Jv0vSmnRIt5ntj3oe9VYnluiytsg4+ee6W5uKkjfusLRe3L1kZJNvV95yd1UZGl7KIxm7LVKd9kFq3bVJU/9CMuE5GfpnWRu84jPiA0nFHFbouV3kxWv/4tOmz6RdrvbpKot8XuX/pslLR/Urg2UC0koIxja6WBu8+W+y1lHXwuW+aK6mdLkvitst0Hc85daqPV111lilv/s32nHUugg8oraTw09ulIn1WeugJ+cAuH+Tj7e7r6qz92GtXCu5Upr5HJtH751O7fqzZp5TXUtbBl8ax/VsKfiPTkJVE8mMOulw0PDonqlkzxisH0H/08yRRzBRaLvoY/XF/GSiDeR36+9XXEZ/Qo8dir6Xig6923UXmE4EuB3DqcW+m/RF6p5yFw81zwsH6oJ40Uu+zEl5LxQcfgLDIzZTQqxzFjvb6A8EHAOg3JRntlRjBBwAICsEHAAgKwQcACArBBwAICsEHAAgKwQcACArBBwAICsEHAAgKwQcACArBBwAICsEHAAgKwQcACArBBwAICsEHAAjKgATfmAtHAQBQFgYk+C4/dwgAAGWB4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAASF4AMABIXgAwAEheADAAyaqZePjpZvWhHten5f9NjBl6MfvvpCdP8Tu025blsqBB8AYECt3LYu2vveG9Ev//Jer6SdtNd9FIPgAwAMGBnV6XBLQ/bTffUVwQcA6HdXjhyeepSXi+wv/ei+C0XwAQD63Y/ffN0Lsr6QfnTfhSL4AAD96uqLzvECrBiyAEYfoxAEHwCgX/X1uV4uT3/0VjRz8qXecdIq++Bbfc9t0e6dm4wlt9+UVbf9u2tM+c6t66KJF5zp7SuW3nlz9NCOjVmuu2yU1861Y8uaaP6sKV55kk1rqsw5bNmwzKu765Zp8blX3XZjVl2+6xJbN90b18+eOs6r1yaP/Yppa7enXXJetGv7fTnPzSVt7rplelaZPb68X3OmX+HtAwBpyI8l6OAqhWIWu5R18O18YJ25+d541djo5msvM69XLp5r6tYtvzPatnlVdM2//GO0qmpe1k3fde/CW00AzJ56eWziBWd47VwPblsf3T37Oq9cs8E0Y9xos497DnJ+sn3Dlf8cXfdvo8zrrRtXmLrxXzs953XZfrd0t736wjOj2VPGmfDRx3ZtXL04Phe3DwnbyRf9k3duLnkPpW7Rrddn7SsfKq7oPs9vT7okPle9LwD0ZvuzP/ZCy1r84JZo3PTJ8faM+XPN9jOH3jZfH38j90hRRn36WGmVdfCtWXJ7dOX5X4q3ZRRmb+ASaG5bKZ8/a7LXh7STG78ut6R/GR25ZTb4ZBQ5aczZ3j7WhpULsrZ18Ejg2W0JMVsv4ZvrumSU2FvQaTKimzDyH7KOv8oJUpEr+CQ05Xg6+LLbLIq23b/S2xcAepNvJacbfLOWLjSvf/7Bn8x2b8En5s67wTteGmUdfNp3Nyw1o0BdLuRmfW336EaX5ws+2ccl4SHl0l5GYG6d3lebOPqMuJ0dJek2UnbnzdO8cve6JIRklCcj2XkzJ3pt80k6pnDPzbabfun58XZS8M2ZdnnW9nf+/ZtevwDQGxmZ6cDSwXfbhlXm60/e+kNclyb45De86OOlUTHBt/zuOVnh5JJymc7U5UKCzw0wGwDybE6eDdp2N0/umXKU13pqUILhgV5GPNLeTlcu7A6RpBCScJNRrFumr0teb16/1Hy1U70SjLqvJEnHtCNNdypVj2J18Mlo1H2/dm3f4PULAGmkCT6x68VfZtWlCb6+ru4s++CTRRdy81121y1endywpU5u7rrOyjXic2/sLqmT9u50o55GdMl0p66zz+10Wymzzw5zXZeUbd+82ivTfSVx29nniGmmTd3gu+rrI7zjybSpLgOANNIEn/xsnnzdtO9Jgm/6pSPNDXfm+DFenR2V6HItV/DJzV6ep7llV3y151z1iM+uztR93HHTZFOeFLxS7j47TJpuTLouWQCzY+tary/dLonuX1an6jZJ3OCTr/p4V485yysDgDTSBJ+8Xv34w+b1T9/579TBd0pOddoAklWTLqlbu2x+dOuMK7PKZQWj7iNX8N008Rumb5niHP+1YWaBiL252+PK4hY7BTr3+qu8PqRcFn245zD+vGGmThaESL08S5Nnj/JaVkq6+yZd14SRw02dhKoE8eZ1S+LzkulKmRrV5+H2ma9/e25SJys2bVs91Sn1MiUrx7c/JqHDGADSyPcbW/Sqzum33xpvpwm+U3Jxi9xwk0id/fk5V9LPq+UKPiE/H2f3dRfNSHsJOlu35I5Z3r5CH1+4y/6r5n0rLk9aharZOhnp2mlcGbXZkahs6xWous98/dtzk7518C2YMyPelulOG/5CP5cEgLTy/ThDMWS1qD5WWmUdfMjmBhsAVAJ+gB0AEJxS/8oyMXPyZd5x0iL4AAD9il9SDQAITr5FLoXgzxIBACoCf4gWABCkvj7vK2Yxi0bwAQAG1Mpt61KP/qSdtNd9FIPgAwAMGvlxh+WbVkS7nt9nRnWycEV+I4uU67alQvABAIJC8AEAgnIy+PzcSlJw8D04ieADAJQPyaV+Db4FlxB8AIDyIbnUr8F3/ojTvIMCADBYJJf6Nfh4zgcAKCeFLGwRBQZfT/gNP/20aP6/En4AgMG1dnxhKzpFQcEn7AFGnvFFwg8AMGjc0BuQ4LMWXTrUBOD1FxCCAID+JVkjj9suPkfnkZ9XuRQdfAAADDadVfkUHHxCHxAAgMGiM6o3fQo+YRe6AAAwWHQ2pdHn4LP0SQAA0P/8PEqr6OCzGAECAPqXnz19UbLgAwCgEhB8AICgEHwAgKAQfACAoBB8AICgEHwAgKAQfACAoBB8AICg/D9yZ7xb/CXD4QAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAABxCAYAAABV0J4vAAAy+klEQVR4Xu2d918VSb7399947n2e53Wfvbs7G2bz3TQzO5vuzO7szs7cTRN3Z8eZcYwgYBYlCooZRCWoiCBIEFERFcUsBsScQSUnJUkWd75Pf759qunTfQ4eFMQD3x/er6qurq7urtNd/alvfavOl+rrykkQBEEQBEHwHr70r389IkEQBEEQBMF7EAEnCIIgCILgZYiAEwRBEARB8DJEwAmCIAiCIHgZTgKuu6ud+vp6bZn6+h5SV+cDW7pgB/XU0d5iS/dWWrv+RRW9RDV9g8daFpwurWkjzeYS+3UPhvJuooO3v7CVOxjK71y1pT0rimu+oOqH9vvylGqN7Cv233ooGMl68YRHjx5RSaO9TgZDeQ9xOdayB2Kk6iXlnP57W+/BU3As3jdruUPFSNWLp5Q1fUGlHfZ6GQx7ywZff9WVpba00cKu619QRY+9ntzR2OfcVj982GMr05twEnAVA7wAA+0bSzx61Kf96HaRqxhtAi7rMlHtoyfDWtbzJuBuNn5hu+Yn4cYD+70OhpH88Fxost/Pk3DkztPVgStGsl48IanYXg9PAsqxlj0QI1Uv1ut+UqzlDhUjVS+ecrjCXhdPws5rg6vD0Szg7nTZ62cg7j0axQJuoBdgoH1mYK372le/bEt/HF1dnba0V176iS1N8eYffmfEn+R8T8KZ0yfo/v1Gam1tpvy8XNt+MJwC7uev/IzDF7/1dds+dwxUh2DC55/Z0szkXrO/BFY+yLanAWtZz0rAff2F/7SlueJs1b9s12zl22vtaVYqe+33Ohg8fbdcoX7f8Z+Os+3zBPRerfdjZeNFe5qVbZeerg5c8TT1MhjOni2mqqpKW/rj2HjGXg9Wvr3OnmYF5VjLHghP6mWgNrGy0v4eDpRfYb1uK1e1jkziBXu6FWu5Q4Un9WLm2rWrVFp609j2pA6sVFdXUVHRCY/KKG6w14WVuQfsaVaSBvm8eCrg3n7rD2ygmDkzwLZvIP7nf/5oS3tWYPTAWj9W7nb3x0XAPQZ8PDs7O7SH4U3efuFr/0nf+uYLdPJkEcdf+tmPady4j+iVl39Kr7/2a/rG179Cc+fMot/99jXOjxcA7NyZy+FH//iQw4//+Q/621//TD/84fccx33VOCf2/+H3v6XFiyONF2j6dD/btT0NVVXl/HAXnThKDQ11LFTz83bY8nki4Lq7u/j6ca0oU90z9qGOWlqa6Qff/46R/o0XvkJtba0cnzTxcyOv2o/8L3zty/Tqz1+mDz94zzjP5cuXeP8f33yDvvudb9H4zz6hN373OkUsDKc9e3bT6phV/Ntgv/UaFe4E3P9aRfTyBqI3UvV45k2i72gfq99u7s9jLcsTAfd17f5xzSe1RhF1gfjt22X0kx//0KmOvvmNr9GZM6c47cc/+gF/gBH/pqNeYSWF0NWP6bOdB7gTcO9mEv1xi35fAPc4cRfRf8TY84JnIeBwH/+lPfsq/rOf/ojOnSsxft/33/+b8SzhnVL5fvf6f3NdWctTuBNwuO9/c9z7i5qIvd5O9H+jiWLP2vOCXYO0CniCp/UCAgPncBvhM3UyTZjwGf3xj7+n0NAgbmvef+8dunnzOn37xa/T7994XRNsZ+j73/uOxrf52NOnT1FNTZVRx6EhQfxu4B18wfHMucKdgPtQ69C8lkwUfLj/+VHPkjUvGC4Bhw4fwt7eHu7wIr4wPJT++ze/5Dzf/MZX6Ve/elUTIfq7g7bh7t07trIU1utWqHv71Sai78XpaT9JJJpXaM8LrOUOFZ7Uyw9/8F0Ocb9oH2/fLuX4j/7r+xz+86MPtW/OBxx314ZA6Kgy0O6cPn2S46+++pIjr/28wJ2AQ919S3vHCir1+LG6gZ+XLecHV4eeCDh8Z9F2og0J8J/Gbeqvf/UL+s2vf6GFr9JLL/2Y0z799GP63ndfpFu3btLftW/NBx+8y/eMNgjf4L///X3j2/Xzl/Xn7xevvszvIZ5B6AN8q8znxrfp3Xf/Sv/z9puc/0+aIIyPX8fnwbf+Oy9+w3a9CncC7sur+9899U1CO/ZBhnPdiYCzgB9Age0FC+Zx+Le//ol/SHMexP/8p7c5VBY48z5ziB8dWPdZ8+GFSk1Jdto/FJScPW3E83Zt5/CW9lGw5vNUwOFF2bIlxfhwqOt9URNUCPESIESDg33paVtY3CAN22hoIXLRyEwY/6nxUljv21UdmtPfeecvtusz407AfS22v4H5c7oefj3WOY+1LE8EHD6Y/v4+tt/XfO0REWEc/kNrLFTd/fLVV5zuNScnm+MNDfWUnLTRdh7gTsBdaNHvLaqI6H9H62kQcNZ8imcl4AA6R2tiVzulI4SAm6iJe5V+8cJ5p/qwlqdwJ+Ag3tTvm3BeD/+Poy5ckX3p6fwAXeFpvahQPSP79+/jbXx8jxw5zHF0uMZ9/JGRFwIOcV/fKU4CDmnqY4649WNjxp2Au97+BdcdfG5Unbn7GIPhEnAIlYVdPT+Io+1obW2hsLAQW/4PP+zvAFqxXrfi37V7+8VGoqttROtK9LSDNfZ8Cmu5Q4Un9QIxi2fh3Xf+agi4zMytvE/VkXqO3LUhb2mdA5VfCbioqEgjzXpOhTsB9+/R/c9HwD49HMrnxRMBB2BUMQs4pKnnX3WsVd3gu4NtiP9XHCNDyoBiftbQ2Ub4bU2EqVEztFXm80ZEhHN+fBfN5/jB93WxPRDuBBzaL/Xu3XFY4PDNEgvcAODFqKur4Xh83FoO8/J2cggB96tf/pzj+JFgFYiNjTZ+aPQA8HH6jab6VRrCZUujONykvUAIYbmKW7fW6UVBfMP6BPrT27rVD9txjvMPFRg2Vb5vPT1dHCohZ8ZTAYcwOzuTGhsb6FVHTxlpL35LF3C/13ozCJH+y1+8QllZGVrv5SuUkZFuejm+TD/9yY/oypVLNmGmwHZw0Hx+kVI2b+Lt1THRlJ6+hTYnJ5G/ny+FmxpyK+4EHBoYJWrwshQ3EfnkO+exlvU4AdfZ0U6v/eaXfI1r16ymT8b9k+M7cnM0sfYB9/CQb+/efA4h4PCxhYB9443XnZ6bBw/aKCBgGvfu0ChZzwUu1rr2gfu3aF284R6/GqtbEkZSwPX2dtMrL/2Upk6ZxNu4vyitsbx+7SrHF0Uu5EYRzyfq43OtPnp6up3qw1qmwpWAK+3Urav/z2FxRBzO9sq64oqRFnCwFKRsTmaLMti4IZF/e3yI0tNS6ZLDGo12KSZmlSHgELcKuJdf+gktiVrE791AdbfRjQ8cRP/8g0SLjmudmjV6fCg/yJ7Wy6qVyziEyMjdnmPcy1/+rHeasR29agUd2F9g7JszZ5atLIX1usHt7v7O3K0Ooh8n6unnm+15FdZyhwpP6gWoe1UCDm1jeHgIp8Mqu3ZtLKe5a0PCw0Np/GfjjLpVFriVK/T6tp5P4U7A4Rl5PVkT/I/0tgfW7qF8XjwVcHhvrAJOGQ0Qok6Skjbo97pyOaVtSWVLN0Z9ghYEcvquXTs4VPVgFXBv/PY1Wx2h056Sksz6APvwjqLOcU58B6z5zbgTcO9l9Xe+J+Xp4fido1zA9XR38sfCmumh1mvp7u6wpT+PfPC+btK1pg8F7e1tdPRIIZ05XaQ1ents+4EnAm64wQsAMPRo3TdY3Ak4T7CW9TgB96xxZ4EbLMMt4IYTVwLuSRgpATdcrNc6hKrzaN2ncGeBGyyD/SCPVL1Yr/tJsZY7VAx3vah21ZruKe4E3GAZ7PPiqYAbbsx+67fLSp+6PoE7AeeOUS3ghKfneRBwQ0nqOftL4CnWsp43AVfdqgm4Pvt1D5aTtfZ7HQzD/eEZiL2l9vt5Eq7Ujy4B5wnJbvwBB8smL5mFiiFh67U/CdZyh4qRqhdP2X7VXhdPwrG7g3vXnhcBNxxcabHXz0CIgBMGZLQJOKxRdaKG6HLz4LGW9bwJOJB50X7dg+FgOVFD+79s5Q6GkfzwdD18ROfu2e/LUy42EaUN0qnaU0ayXjwl7YK9TgbDoYrB191I1UvulS/ofJP9HjwFS9ZkXByc+BgMI1UvntLbp4s4a714yiUNdKit5T6O0Szg7jR/Qceq7XXljrI2EXCCIAiCIAjCCCICThAEQRAEwcsQAScIgiAIguBliIATBEEQBEHwMr6ERQqxFpkz5jTn/ffumeONjm3kqee4eb+7MvrT9GPcle9q2z2uzuE5ruthsOj3hLjr8vrTUHfmff31aM1vTbPu7y/v6X4La1r/NalyXJfnCtflKR5Xjuu6Gwh3+Z/mt7AyuN/CfMzgfouBeNrfwn6+x5Xjuu4Gwl1+z34LK9brMz/j1rwDl9f/W9jLc3fMQNjLejz953BVD48rz9UxA+/rT3N+xp/+t7C2N9b9A5f3uN/C9bZ77GU9noF/i8HTXw+uy3P/W/TXozW/Nc26v78882+BuOv6cFWe6/NYj7duu8de1mDKcV13T8KT/Bburs91HQ1c3pP8FgPhuu0XC5wgCIIgCIKXIQJOEARBEATByxABJwiCIAiC4GWIgBMEQRAEQfAyRMAJgiAIgiB4GSLgBEHwGqZNHW9Lc0d3d5ctzRNul92ypT2OqEUhtjTFju1ZtjRBEISnRQScIAheg1XAhQXPoWk+/WkB0ybS1EnjjO2Y6KWchrjvlM9otbbdX9ZnFOCn76uqLKcZAVNo545sFnA4ZvZMX6qsvKuFPnTj+hWaPPFj8p82gR496tMEWyj5TP7EKAv71q1ZyfHz54tphv9kmqmVl5uTyWW2tjbTrBlT6dDBAvKZ8in5+X5uHFt66waHPT3dRpogCMLjEAEnCILXAAHX1tbCYPvObd1atjgymPr6HlJdXY1T/phVSzisrLjDIcSX2hcZvoDFFOIQcDh+UUSwYYGLigxhYYb43Nl+Rvz0qRM0ZeI4mjNrmlHWrBk+XDb2IR8EHNKXLA5jAeczWT+Pfq0hNNd0LFifuMZJiAqCIDyOYRNw9Tfv0MnYNCqJz6LzCdlUvC6DzmzZacsnCILgKcqaplgfH2sIK7BkcSiFhcw1tvN25Rjx6JWL6YwmsNT2lEnjKGljHMch4FauWEwnjh/WRGEpp21NT6am+42aWPuYqqsqDIvblcsXaGtaMs2b42+UtUo7FuGGxLUcQsD5auLw+NFDLOAwnItyLl08x+WsWq7nVxw5fOCJh3wFz0hLTTLip04ep7WxusW0YN9uIz1+XQyD/RvW67+lOb2y4q6tXJAQF0Pbt2VwHBbVLSkbbXnA7rxcp22z1XVN7AruRCCuzod4asoG2rhhna0sQRgWAXd0XRpdSNzmkqLYLfSor78XLAiCMNz09fWygFIWNytVmkCzpg2WrWmbKHBuAMcxfGrd7477mkicahqOFYaWlE2JHC4InKEJ7DUc37VzG4cQ3ugEQKyr/FcvX+TnBcPl+XnbjfRELR/Chw97KXNriu08F84XG3EMmSO8e6fMEOYpyetp+dIIp2MC/CY4bavOyMmio8a5GuprDWEnCGZsAq6hrlzrddY50VDveeN2OiXXJtqsHI+xP/yCIAiCMNTAfxH+iUocdXV1sH9jSNBsighfQPPnTef0QIdFFUIPocp/+NB+DuPXRXOohuW7ujopbm009fb2sBXPd+pnxjnhX4mwsaGO/SwxBK/KxDGgo+OB03XmbNtKmx1iM2dbBmVnbtHifTTdbxLNmj6V03EcBL/5OGHs4iTgqsp1Z1pXVA6wz8z5BLtgc0VvtzjsCoIgCMNHTXUlh/BtVILs4cMewxeyrOwmTXf4K65YFkn79u4yjkV+WOKKTxc5lTlzut26CivblcvnOY6JKi0t9zleuH8vD5ljaF+VqY5pbW3R8jUxKg2+lQghMuvraqi1pZna29ucfDcFQeEk4MrvXLVl8GSfoqu1zRBoQR9NopK4TI5Peus98vvLP2hvVLyxv3CFax+B0UrIgtl0924Zx9VLOs/RKwPbczL4JVX7ik7oJnQrRQ7TOpjp6JV5grnheJ7Iykyl9C2bKCsjlbfbH7QZ+xaGBVJBQb9/yuOoq63mIQfrTEUMkZi39+TvsB37vAH/LIQ7crNsvx2GZzCkgnu1Hjfa8eSen9Vwk6eTDjCRwZo2ENbf2xMePGjl9mP2DB/bPkFwxdKoMEqMX21LF3TemdDvPwvenZhDi9c4i/mRZkgFXFNFtVsB95dXX6MPfvMHY/+hFf0OpWOBgGkTaHWMvoQBPsrmfX2P7B8cCDiY2OGzo0zzmLkGAadmtEHAwdx/7OhB3j5bfIrDZUsWcqg+BFhWQcUz0jezOIDZ/+aNa5wevGAWNTXdo/r6WsMZ+1kBsTV10icMhFx3dyennzhxRBdw+3bzbMDr169wOj5S8BfasT2Tt7H0A8KOjn7hF70yyhBpaKCUgMPsv+bmJt63N38X+6ygN96m9YTV8MiyJeEsojE88qyEgCv2F+QbDvvqt8MwEPy4lIDDkM/lS+eps7ODgubPNPLh+vH8oD6xHbkwyBCEGS58d7wJPCeZmtjHc4ChLQxn6WK2h/fjncD22tjljvzj+BgMOyG9tqaaKsr1GanxcbqT+LKocPZPQhx5bt28zqIL7x+2GxvqDd85nKe56T7/DmYBh98GQ12XtN8DIhO/gVqyBL8LljBBHBMaECZvSqQzDstOe/sDmjt7Gr+DKr/5nj1hv8MRH5MtrPsEwcq9xnrjnamtqbLtH8s8aO+mN/+pT0ixMn7mblqy9qQtfaQYUgHXWlvvVsDNfv9TpyHUg2NMwK1cFskfUTi1qob/8MECY7+yLChTu7LAJSbEUkT4fI6jYYeAg7DBNj5WEGJqSQKILzTg6gOgQsyIU/Hw0ED+0CCev3sHHT92iAr27uKlE+41NjDm6x5uLl4soYsXSnj2Hz6KtbXVGlUsPnCfuG4sEaEEHFixLIKHOxBXAs4stvx9J2gf9wMcNws45WRstsB1drRziHOahyngizKSAu522U0Ozb47+J0RNwu4rWmbeR+c8FU++Nxg/bGzZ3VBDzq0+zx6pNB2Hm9DCTjEVy6P5FmfmDiglhXZlp1u/G7HtWeqoaGOpvtP4ue6p6dL66TU8ExQ7FcCDh+yGzf09k0dW6QdqwQctpWf1IMHbezkjro2CzgMg+XvzjU6Fv6a+IYTPOLI29hY5zQjNiJsPj9zahsW+IOF+4z8Kt1Tyu/e5jA9VRftgjAwfdyRedbtvbfw9rhMrV1wHrbG9wHCzpo+kgypgANmkTYQtw6fth07llAfHDOdnbqYcAX8JczbamYTHiqz8HjQ1mrEIRjdLU0Aq401TeU1+2SMBK6uzR2hwXOcj3UIMmCeoq/qyBxC/CIOfxNruQP9Fs8KWHpUHNdqFpTm36ij3dkZGihLphm1ZMZoBZZUa5pC/dZDsVgufJKsaWbMz475Pd+WlW7E0WFz5dcEgWhN8xQMo1rTBN0nzc9HXzg5PGSeMeMX/56hJiRA5KtnBGBkA6MViDc01BvpG9ev5QkQiGMkBGXjvewyvW/79ubxMjOIF+zdzYtBI46RlMIDe41zq5ET5TaDzuiBgnyOYymc4jPPYrjO/gwKOn8dn02XrvX/9hB11jwjjZOAa7pfa8uguNfomZn1aPRmm1hzhfU4Yehx9YEYTZgbXGFg3Al54dkA6581TXh2LAzVRy0ABNOjRw/Zkg3R23T/ns3qOWeWL6Uk6zNCebh8qj6EfudOmSEAy0pvGp0ifYma/qVgzDNSKyvLOTT7m+F8SevXcVnKvQFpWx2iEWmZGd7t7iAMP7ZlRDo726jpXq0TnSb/osfRfr+ZzsVn2QSbmaNx/T1RQRAEQRguThw/Ykz+Cg3qt9hXVZUbQ89wVUCHV1nUsBiv8hErOes8WgS/U4TNzbqFvKamyvBfxEQ1a8dZCTh0oopOHOG4WlZkS2oSCzf8g4eRv+KusZ6gIAyETcANFUVJ2bYlRQ6t2sQzVa15BUEQPMFs2RiI4V653vqRFp5f4Bushs6xGK9akHd9whpj8sd+y2x3+FKqRX/hf6ssbZuS4o1JaPv27qbNmxI4fv5csTFpSJUPNievN7YL9ubxORFPT+v3VcTfqCHExAJVhgoFYSCGTcCBR1pPpq2+kZqraqm7ze6nIwiCMBgg4ObO8mMBFRQ4Q59sosWXLg7jmaYYqsRSGkrAYeJKyIJZvB6YmogAHyZYPDA5BtshQXP4w4r9Z4tP8qxsfJCVrxNmQcPXEufEPqySHzx/Fset1ycIgvCsGFYBJwiCMJSYZ1hj5qaf7+fG5A6IO+WfpASc+kuilM0beLkOzMbFNoSfeSV8OK3Hrl5uzNRVS7hAtMXGLON9yuqG/0D1ZD06QRCE4UQEnCAIXoMaQoWAA1gyxizgsLI+0q0CbveuHB5Ge9jbw/tXO2YfKkE43W8iO6IrAaf+ugiiDWu46Y7vuoALD5nLcaSpv0wSBEF41oiAEwRBEARB8DJEwAmCIAiCIHgZIuAEQRAEQRC8DBFwgiAIgiAIXoYIOEEQBEEQBC9DBJwgCIIgCIKX8aXeni4SBEEQBEEQvAcRcIIgCIIgCF6GCDhBEARBEAQvQwScIAiCIAiClyECThAEQRAEwcsQAScIgiAIguBliIATBMFrwJ/H93R38h/JW/eBnOyttjRPQJnWNCuNDXU0eeLHtvTHcbvspi1NEAThaREBJwiC1wABFxY8h+bM9KWpk3QRh7jv1E+prbWZBRxE1p7ducYxfr6fG8LLb+p48vedQK3N9+nCuWLalpVGUyZ9TEcOFvD+rs52zr8+fjVvmwWbWcD5TPnUiK9eFUV375RSY30tbx8u3EeBs/2M45WAW58Q63QvgiAIT4MIOEEQvAYIOIQLwwI14ebD8amTP2FBhTgE3N78HZqg0/OBuLUradHCII5DUN29fYvjKcmJNN1/MmVuTTEEnMoTGb7AiKt0s4CboonHiPD5LPia7tU79usCbkbAZArwm8jx6X6TDAG3IHC6UZYgCMLTMmwC7nZRCZ1dl0HnE7bRhcRtdC4hm46sSbXlGytsXL+W7jfWcbynq4NmaB+OuppKY/+mjXE0c/pU23HuKNyfb0sThNHOdE0YQUSdLDpCDfU1bD3r7GijttYmCgqcaQyhPmhrNo6xCriM9M2G9a6r4wGHZgGn9oWHzmNrnErHsbD0qTwQcIivXBZJfj56Ph9NTKp3c2FoIAXNn0l3busCLmTBLKMs4dkTFRlixLdnp9Psmb4cT01eb6TPmTWNyc/bzr+dNf3a1Yu2ckHgHH9atmQhx0+fOkbBbn7rhLgYp+32thYjjutBhwBxdT7EQ4Pm0DytfGtZgjDkAq67o51OrUln0eaOi7n9jeVYAY3/gnnT2TpgNCQu/G4g7hD6+YznEKIubs1KfvFjVi428uXn5fJHAz38iLD5xnG+Uz6jmQFTOA6RiA9Qfl4Oby+KCOaPXn1dNa1ZvZz279vN6VGLQmjp4jBqbblvux5B8DZuXLtsSxss3V3tbj/WnoAP8fWrl4z3Uhg5VkcvodrqCrbKrl61hNO2btnEISytsJYu1MR6+wNdTC2K0MX+1Emf0OwZPoaomuZokyHsF0cGO52j7NZ1ra3W212gxBfKCJo/g4fYlywOpYBpumVWYX7Grl25aHwbcrdn0OFD+ndyndb+q+sVBDNOAq6t5Z7tv7YUrc0NtoNdcWhlkk2wucJ63GgHYkv3h7mhNQBzOa2jvc2Wr/l+I4d3bt/iD8Chg/u0npkPv8CBcwMMZ2uUdfP6FSo6fljr9etWgRqtkVLldHXqloWK8tt8DCyAUVqjM18rAwJOvya9QYqNXsplHyrcZ7seQRAEbwbWMQg1NfwN0YR2Nm/nNlq+ZCGFhejtsa9jGB7WN4Qq/9a0ZA6b7unfwH17dnLY/qCV0rckGQLvbPFJ45x1NVUcYngf1lpVFkIcA8KC59L+gt2MOs48OedBW4v2TW7SxN8t4xw4Tg3ZC4KTgKu4c9Um3BTYZz3YFUqgnV2XacSLtXjBkkQ6uHyjkVZ16Zrt2NFMudYDCw3WG4p9e3bRhsS1hp8NwNALelppqUlGmnrp8fLO1Xp0CXHRhtUOFri1q5ezL87ypQuN9Lmzp9GsGfpQ7LIl4WyRQ3yKVhYag6zMLYaAmz8vgEP0CmNjlhk9vmeKxQoJC6GKwykcQ2OIuxK7VuDEjp6wNd0K6sGa9rxRV6t/AIYK1TEQPGOo6v/c2VNGvKGu/9keDq5eeXKL4WgFllRMHolcuICH3xPiVnObg4kvaggcVrbYmKVUW1NJ/tMmGMfCr3JpVBi3FwcK8o10jKSg7TWfB1Y5ZV2DEJwZMJnjGA1BOw7rGtp3ZZlzBax6y5dG8DVjCBbCs7urg4dylVVwqKk3P+eWtliJRrwL/D449t+8ccVWzmhkTkQhXbmu+7WCaUHPn4HDScCVDyDgsM96sJXmmlpDoAV9NIlK4nQRN+mt92j2+586WeAOrugXKsLYBY1esNZARYQFUkvTPU6DgzqHWqO4L38n5W7bSuV3SzkNQvPE0YMcR4MXt3YVx9HoqTIx7Hyq6AjHd+/K4V7trZtXKScrjS2R+VpawroYLgt50EjDkR1xWD4PHywYtgbTUyDoVRwfH+VXFRo8R3sXb7FVVQ3jJCfFcx4MD125dJ7T4H+FEA71GEI6V3KKqiru8PFoiHdszxjxe3wS8CFV9w03gKOH9vP9KCsKfOOWRoVzfJkW7tqR7XT8pg1xdPTwfo7jowuBFbMqip81DHei04PnRFmjw7VyYenGRxgdH/jSnTl1nE4cO6TV5WzOAzcF+NkhftzxbILolYv5GcPQGZ7TubP9KDRIP2aogbUeYd6OsTe6ITwZ2y1L7ly6WGJYGYF52FZ1oGMcQ9AwKFjLG01AvFVU6t8jRX19M735zwxb3pFkSAVcY9ldtwLuL6++RhP++I6x/5AIOKFHF1vopULIqR7xtswtlK2BjyoaCnxYr17WhQmAwDMEir/e08Vwhtq/dFEoCzfE49dFa+iOwyUOawgEnMqrnIghbhDCD1Afnn56P6qnQQm4SEedHDl8gKor73Kc10FzONonxscYDvQA1z7D4QMJSwFEDuL7C/IoSqsXxOHLowRPdVW57dzPM+yGYLKy7tyexSFEF+pFLR9yVKuvpA3raJ5jOQ+A/XhOEPo7Jiekpmwwhq3Wxi4nX4c7Aj5esEAosccWF+04lIl63Zad5jhmhSby5hnngFtCUOAMam5q1ERyJlvAlTXPsKg7Jk4MJWoYrqz08RZoQQCwFGIYWW37a8+uWv4GTPefZMTVxJ5tmakcV23MaOTshUq3Qm1DWonbfSPBkAq41voGJwGXOD2UNswMZwE38a13Oa72F67YaDteGINoH0UMLcFXDx9CCCf09tJSN7KAgz8KxIxZwGHIGBYQWNWUgAMnjh2kew21PExx68ZVFmcQcJjhi6EIWE/gTAwBh8kkp4qO8kf6TtlNpzW6ViyNYLHkybDtcIF7vtdQR4cP7uNh5ZCgWdTSfJ+vaZYmetG4ws8GFiQIBnUcRAIsirBmwuqDCTA4DgIODXZHeyvXs5oNhzq0nvt5JmZlFFXcLePnBvWDSTyYhANRhpmn8+b40f179XzPmP0Ji5saCgJY9w2z/bZowq2l+R4Pk6mPE56rSu05RB1CwMHXCPWNOoJoQx4M68OypwRcTPQS7lDgfNiGJQwuDEoUQsDBcoffEBY4pOE3sN7X04IyW1uaDEd7QXgcGC7OyU7ndwedZFjg1BAv3rOMdN33Ty27A06fPMbP8sEDe2zljSZOl1RQQupZp7TkzPP0D98dtrwjicUH7ppNuCmwz3qwK8zDpAMBsWc9drSDoRiE8EcqvXWN4/hYwB8OAkOtI4WPBk9W0CgrveE0PCjo3G+sp8L9e1iEWfcJgpnjRw/Z0oSxBRZ/hpBHO4tJX7CCog3BkCHiZn9kAD8v1e7CN67B4TcM9u/L49BsvcKEA3Qc1LYqD8P56DQivsbhsgEw6gAhhI6msqYrCy2Aj576HggjQ1Gx8+jEkaLbtjwjjZOA69Ee2I6ONpt469QecE9FBCxrVrFm5Wxcpu24sQB8vSDK1ExSvNBKwME3qfTmNU6HRQA+WeYXuGBvHvv3WMsUBEEQ3IMJX5jEpbYT4nXrM3wglzjcCpSrQYljJiksqEpQYd1AZY3H4s8qHRZt+GNCFBafPsFLlTQ5Jgth+FydTwk4WNV35epD/sqHbNPGeCcBp3wkMYFBHS8I7hjydeDA4ehkm2hTnIvPosbb3uV3MxRsTFzLPUAINeWvhSFCJeAw3RxpcEjPyU7jGaIRjl4cZldiVpJ60QVBEATPgQUOIf5GzZyuOslKwAGMkDTf7x8h2puvLxvCaJ1vtSA7T6zStmtrqniVAYg8zOq859ivUALOjBJwWMYE4hIddAhApNXWVIiAEzxiWAQcoz3YsMadWpvOFrdj0Zvp7ul+PyZhcMAp25omCIID9oHrX9Xe0xEDQRAEb2X4BJwwtDiGXQVhLBM4158nbmCpFyzfgQWo1QxRCDjMIt6Zm0U7cjPZuo2hrayMVJ7sggVZzQ7ZgiAI3owIOEEQvIbFEcHsSwR3Aog1/Ln9LjiTOwQcZqXCJwkCTv2hvPl/JEXACYIwWhABJwiCV2H+z14sKdLo+PcONTkIa9thHbYaxxp3tdWVvMQIfEmxvEpPt/w/qSAI3o8IOEEQRj3pqUmUt9P5XxkEQRC8GRFwgiAIgiAIXoYIOEEQBEEQBC9DBJwgCIIgCIKXIQJOEARBEATByxABJwiCIAiC4GV8yfq/p4IgCIIgCMLzjQg4QRAEQRAEL0MEnCAIgiAIgpchAk4QBEEQBMHLEAEnCIIgCILgZYiAEwTBa5g2dTz5TvmU9hfstu0DudszbWmeUHhgry3NSmdnB02e+LEt/XGUl9+xpQmCIDwtIuAEQfAaIOAuX77A4ZyZvrR/Xx7lbNtKm5LiKXXzBhZwYcFzqaz0hnHM1Mmf0JRJ42h3Xi7FxiwjH00AbknZSI0N9XT50nmKCJ9Px44e5Lx79+yi2Vq5J44dpgMF+YZgO3L4AGVuTeHtwv17tDw+HD918hhfx7o1K6mp6R7nPXqkkHJzMujQwQLOAwF3+NB+vk7r/QiCIDwpIuAEQfAaINwQLpg3nRaGBnLcd+pn5OfzOcch4NasXk5TNcGmjolbu4oWLQziOARVXV0Nx+PXRbOwWxwZYgg4lSckaLYRV+nNzfeNbRwXsmAWdXd3Unv7A067d6+Bw6jIUPLzncBxiEFlgZs7288oSxAE4WkZVgHXXFVLpzbl0KHoZCo9ctq2XxAEYTBAqEFEhQbPpe3bttIUhyArv3ubpvmMN4ZQW1uajWMg1BZHBHMcxwbPn8XHPXrUR52d7ZxuFnBqn5/v55oQ/MQpHcKN41oYuXABx6f7TzYE45SJ4ygseI5x/HxNaCoBFzR/ptO9CIIgPA3DIuB6u7vp0IokupC4zYkz67bSrTEq5CLDF1BtTRXHVy1fROdKzlCA3yRjv/+0CXSy6KjW859jOxZDNEeP9H9g+ulzkTYwwQN8RO7fb7SlCYK3sTpmmS1tsGD4MyF+NT140Gbb5wlbUpM04RhDaVs22fYJI8OCwBlGHFbbI4cLqb6+ltYnxBrpaGfBwcICbpOt6RVu/Bkztm7mdhpxdCQg3K15wPKlEU7bq6OXGvG8XTnacQEchxX5+PFDHI9esZiSNyXYyhIEm4Crr71L5XeuOlFX4/qhdUX5mUt0IcFZuFkpXLHRdtxoZ9aMqbRiWSTH83fnOu3r6+u15YdT9cOHvZSyeT03DGhMdu7I5n146Qv2wYm7j3v6SxaHUUNDnfPx+/fS1In9w0go68b1q2yBwDFIO37sEIu2vr6HmqA8LQJOEIRRydb0zcbwN9pCdJ7R/oWHzjM60rGaaELoO+UzDpG/p6eb20ZsT3fkmzXDh0O0m2VlN6lPa09DFsymgGkTjfMpi+yJ44c5faXWaVdl4higylDHYJ8SiDjXrOlTjXR17Tiuu6fLOEYY2zgJuLrq27YMitqqMluaK86sy7AJNlc8Mj24Y4FNG+LYmfr+vQZ2okZaTna6sb+3t4fDaVP1fejBI+zq6rQJuNOnjmvpHaQscHk7t3EIh2pVHoZrzP47KH9RRLBhgYOIQ+OEuGpERMAJgjCaaGlpYsG2KCLIaA/Rpqr9uTmZFLxgFsdnBkymjo4HRnuo8hcWqhnKenu7MXGt7TygpqbSiPc62tb1CWtohv9kmjl9ilOZAH6ZEGlKqAF8I1S8u7uLGhvrbecRBIWTgIO1zZrBk30KiAIl0II+mkQlcZkcn/T2e/SXV19j1P6zW10vAzBaaWtrpb35Ox3bfZQQF2MIKHDh/FnaYGkY8PJfvXKRsrPSuAeJPEiHv09vr34sZrtdu3aJ4+bjkX740AFTWbHcIJiXWdibv4tSUzYYFjmwbs0qp2sYblAvHR3t1N6uD1VhNp/atzAskAocy0VcvFBiO9YKhkV8tAawpbnJKR3D1ebtPfk7bMc+b/hM1oX801JcfJLDzIxU276xjLJ6Z2xN0d6JLNt+BT78eG+s6c8jGLrLythiSx/roB3Nz8vVOro5PCyOtH178yhDa1MRP3HsEK1PXMPx1JT+0SHMOt6UpA9dmofCIari1kU7nWNr2mbaX5DPcYyG7HJ0qtO147ZrbTHiCXGreWjefJyZlM39bTHeV1wz4vD1VNc6lOC5bm3VfUVnz/Ax/DsxE3u6f797D2Z3o21FfNeObPLTnjO015il7e87gd0MLl0sMSyFuAd/38/pwgX9e+WtvPnPDGpp7Rf72LbmGWmGVMBh0oJLAffWe/T5m3+j5DmRxv6DK3ULkzC2wVADhhjQYBw9coAtiwX78ngoQQk4fJiuX7/C+dGYYSZiVWU5i9QZAXrP1iyGI7WerRJpiVqDPW+2H/sfwqk8XhPO2IdZjIsjg7mxwSxFNTyCa4Gz+rasdO65W6/3WaEEHGY2QmRjWAfXs3fPTrYYrI1dQZs2xmsNcAs3mmmpSVw3mBlZVHREu/40uqg1qvCpwZA9rAghQbM4jg4BnPO3aOVaz/u8g0kCsJKobeW/VFOtWz/Uh/aG9rxg8oDq1JTe0pcVwczVObOmaXWgW7/NAg4zRhGGhczj8Py5YhZweDYPFu7jNHQwqqrKHefvMz7aCvguKUu42doy3CgfWdy3dZ8guEMJOFB8pohD+OKpWdiKlct19x+AZxy+g0rkQfjhPZk/N8CYiY12dSB/6+cdd2Lt7XGZdOvOfVv6SPHMBNzs9z91GkIVASeArMwttDM3S+vBJnMvDwIOa3Sh5woBh48gRIcScAAzCtVkDyXgzGIL4kz1gCHg1sfrH/m7d8s4NFvgsDQEQqzhpawyGOZWvdGRQgk45XgNS+zNG9eM/UocwK8yyOScjXTliwOhAqGHOASccu6HCEIjjHipab00b+CYJlRqNDGuWyr6eCZoVGQI78OM0phVSzheXV3B4dmzp4xjle9S5tbNhoBL2hBnCDjleK4+VtEro2wCrvDAHsMHCUNtm5MTna4Pz67yY3qWAg7r2SGEz5V1nyC4Qwk4dHpVPCM9hUOr5dk8UhMRNp+38aybh33VOxa5cL7tXN6EOwH31seZdLvCeYRnJBlSAdfb2WUScJONYVMIOOsQ6lgVcFarTmOj8+SDscjD3h72S4F1CcOpUyZ9zIJOWeAw9HHt6mUj/0rHZBBY1GY6BByAZQ4fTQxhQAhChEHAYVHXs8WnWOzB/A8Bpw+36iIJ+eCnosrBel2wwlmv81kCP0ncC0QJQvRukY7r2pGrNSJlNw2BYO7pIg3PGEI4V0N4TNPKgoDbnZfD6XW11Ybl6s6dUtu5n3daHEuEmB3Aa2uqjTgEHkIMz1uPtaLWcLNS71gr7mmAgF6ofeis6cMFftvolYtt6WMdLLzc7phNfPt2Ka/dh3hlxV1t+xbH0T4oNw5QXVVhDAGafeZgla2q0jsH8K87VKgPiZqfxZKSM4zaRpuGsK21hRd0RvzBg1Zj6ZoSRycD1lN1HN57lXe4UO0l4ghVPCJ8gdH+4dqWLA419q1ZvYL34bsFi7XKB4s/3FcOH9zPQtBcnrdiFXHYvlb6fPmJWyYxuJ9tKpMYng4MgeGFxUMNIYEhLFhP0HCgJ4OhRAwD4m+C1LZajwrHsl/BtAkUtSjUVvZYBPVhdv4VBEFwh3niATpucD1AJw+dnarKuzaxgYlibW0tHMewuRIqEGqxMfrSH7DYKzEYHjrXqdNntuBXVpZzaPY/xvnQAYWFXeVFmrJ6IY61Ds3XJAhWPFpGBGnWfO6oOHtZlhFxAYZn8PIrS0pszHIWa0rAIc1sioaFBAKlp6eLEh1DgLBIqRmsgiAIwuPBUiFqCBx/kaba2w3r1/IEB8Qh0pAeFKhbsyH4sC4n4hBz5vIwGQEhrG+wzsHCdv+ebpmBVQ+h2SKnBNyxo4XchiOu/pWjrPQmCz8IyurqSr4GWAIDHd8JQRgIm4AbCm4dPm0TbWNZvIFO7SUPDZ7DXL92mVePx4trFnBYMR4zevDC78jN4iGYSxfOUUNDPS/xAWf+mZqoU2sICcJYA1YLaxrAWohBga4XT1XLPwT49a/T5Qr8J6qKmxdxBeZZecA85C4836ATrJ6bzZsSDUF14XwxhYXoVi7z7HcQE72Erly6wHHM0FRrd27LTjeeE7TFmM2KJUNgSdu8KYGtcpi1qcrBULzyw1wTs4yKz+izws1/q6YscLAM7szVl4oSASd4wrAIOEVrbb3xV1plx4pt+wVBEAYDXA+wxAImLjQ01NKMgMlap2g2W1nwMd6hfQAxqxhDY/jwwv1gWVQYrVqxiAUclkjAQtiXLp3jMGnDOmpq0iey4COMcmFRQV5YXmAxwUQHDGnFmv7hAZ0vWNPh9oA1F9VEClwfrOSnivSZqMPtxyQIwthlWAWcIAjCUKIEHOIQaDscM0iVgFMuCgCzcWF5UYtkKwGHOBy1MVsZcbOAQ7hqxWLOq1walIAzX4fahqhTa3vBeRszmFWeDYlrjFX9BUEQhhoRcIIgeA1mAYflPrB+HyYDKQG3fVsGL50C0bZsyUIenoKLwupVS5wEHJZNwEznqEUhxlIySsBhTT3khasC1txL2mgXcFiKZWlUOLs8YK09LDmCdPi1qpX94eeKmYfWexAEQRgKRMAJgjAmwd/VTXOsMO8KDH8ucOtXNzBwmodPqzVdEARhqBABJwiCIAiC4GWIgBMEQRAEQfAyRMAJgiAIgiB4GSLgBEEQBEEQvAwRcIIgCIIgCF6GCDhBEARBEAQvQwScIAiCIAiClyECThAEQRAEwcsQAScIgiAIguBliIATBEEQBEHwMkTACYIgCIIgeBki4ARBEARBELwMEXCCIAiCIAhehgg4QRAEQRAEL+P/A6NM1h1lHHKdAAAAAElFTkSuQmCC>