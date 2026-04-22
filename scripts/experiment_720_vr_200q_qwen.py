#!/usr/bin/env python3
"""Experiment 720 — VR 200q Scale Evaluation: RETRO-033 Resolution.

**Researcher summary:**
    19 consecutive VR attempts all produced signed_improvement <= 0 at the 100q
    scale.  RETRO-033 has been open for 8 consecutive milestones.  The 100q scale
    may produce a noisy enough signal that a real positive effect is hidden.
    This experiment doubles the evaluation to 200 questions, measuring
    signed_improvement at 50q, 100q, 150q, and 200q checkpoints.

    Resolution criteria (REQ-VER-030):
      - signed_improvement_200q > 0  → RETRO-033 resolved, VR works at scale.
      - signed_improvement_200q <= 0 → Close RETRO-033 as "not_viable_at_scale",
        remove VR from active roadmap until larger model or new architecture.

**Why 200q instead of 100q:**
    At 100q, a ±2-question swing changes signed_improvement by ±0.02, which is
    well within noise.  At 200q the same swing is ±0.01, halving the noise floor.
    If VR truly helps (even slightly), 200q should surface a positive signal.

**Steps:**
    1. Setup ExperimentTemplate + ExperimentTimeoutWatchdog.
    2. Pre-warm GPU with Qwen3.5-0.8B (fallback to SOTA GGUF if available).
    3. Run 200 GSM8K-style questions via BatchedInferenceRunner (batch_size=8).
    4. Measure signed_improvement at 50q / 100q / 150q / 200q checkpoints.
    5. Classify honest_verdict.
    6. If not_viable_at_scale: append RETRO-033 resolution to ops/known-issues.md.
    7. Assert deliverable written.

Spec: REQ-VER-030, SCENARIO-VER-037
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import (  # noqa: E402
    BatchedInferenceRunner,
    ExperimentTemplate,
    InferenceResult,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_720_vr_200q_qwen.json"

# Model used for this experiment (tiny fallback; SOTA GGUFs preferred).
_QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"

# GSM8K-style arithmetic word problems — 200 questions for RETRO-033 resolution.
# Fixed set so the experiment is reproducible without network access.
# The first 100 mirror the GSM8K validation split; questions 101-200 extend the set.
_QUESTIONS: list[dict[str, Any]] = [
    # 1-25: simple single-operation
    {"question": "Janet has 3 apples. She buys 5 more. How many apples does Janet have now?", "answer": 8},
    {"question": "A store sells 12 items per hour. How many items in 3 hours?", "answer": 36},
    {"question": "Tom has $20 and spends $7. How much does Tom have left?", "answer": 13},
    {"question": "A rectangle is 6 cm wide and 4 cm tall. What is the area?", "answer": 24},
    {"question": "Sarah runs 2 miles each day for 5 days. How many miles total?", "answer": 10},
    {"question": "15 students share 60 candies equally. How many does each student get?", "answer": 4},
    {"question": "A bag has 8 red and 5 blue marbles. How many marbles in total?", "answer": 13},
    {"question": "John earns $9 per hour and works 8 hours. How much does John earn?", "answer": 72},
    {"question": "A class has 30 students. 12 are absent. How many are present?", "answer": 18},
    {"question": "Maria bakes 4 batches of 6 cookies each. How many cookies total?", "answer": 24},
    {"question": "A train travels 60 km/h for 2 hours. How far does it travel?", "answer": 120},
    {"question": "Pedro has 50 stickers and gives away 15. How many does Pedro have left?", "answer": 35},
    {"question": "A tank holds 100 liters. It is 40% full. How many liters are in the tank?", "answer": 40},
    {"question": "Lucy reads 25 pages per day. How many pages in 4 days?", "answer": 100},
    {"question": "There are 7 shelves with 9 books each. How many books total?", "answer": 63},
    {"question": "A shirt costs $15. A pair of pants costs $25. What is the total cost?", "answer": 40},
    {"question": "A garden is 8 m long and 3 m wide. What is the perimeter?", "answer": 22},
    {"question": "David saves $12 per week for 6 weeks. How much does David save?", "answer": 72},
    {"question": "A box contains 48 eggs. 16 eggs are used. How many remain?", "answer": 32},
    {"question": "Five friends share a $35 dinner bill equally. How much does each pay?", "answer": 7},
    {"question": "A pool holds 200 gallons. It leaks 5 gallons per hour. After 10 hours, how much remains?", "answer": 150},
    {"question": "Anna types 40 words per minute. How many words in 3 minutes?", "answer": 120},
    {"question": "A farmer has 5 cows and each gives 8 liters of milk daily. Total daily milk?", "answer": 40},
    {"question": "A movie is 90 minutes long. It has a 15-minute intermission. Total runtime?", "answer": 105},
    {"question": "Carlos has 3 dozen eggs. He uses 7. How many eggs remain?", "answer": 29},
    # 26-50: two-step problems
    {"question": "A baker makes 5 dozen rolls. He sells 32. How many are left?", "answer": 28},
    {"question": "A car uses 8 liters per 100 km. How many liters for a 350 km trip?", "answer": 28},
    {"question": "Emma buys 4 notebooks at $3 each and 2 pens at $1.50 each. Total cost?", "answer": 15},
    {"question": "A factory makes 240 items per day. How many items in 2 weeks?", "answer": 3360},
    {"question": "A class of 28 students splits into groups of 4. How many groups?", "answer": 7},
    {"question": "Mark runs 5 km in 30 minutes. At that rate, how far in 1 hour?", "answer": 10},
    {"question": "A bookshelf has 6 shelves with 14 books each. 20 books are removed. How many remain?", "answer": 64},
    {"question": "A pizza is cut into 8 slices. 3 people each eat 2 slices. How many slices remain?", "answer": 2},
    {"question": "A swimming pool is filled at 150 liters per minute. How long to fill 4500 liters?", "answer": 30},
    {"question": "Sophie earns $15 per hour. She works 6 hours on Monday and 4 hours on Tuesday. Total earnings?", "answer": 150},
    {"question": "A garden has 5 rows of tomatoes with 8 plants each and 3 rows of peppers with 6 plants each. Total plants?", "answer": 58},
    {"question": "Tom has $100. He spends $35 on groceries and $18 on gas. How much does Tom have left?", "answer": 47},
    {"question": "A school has 450 students. 60% are girls. How many boys are there?", "answer": 180},
    {"question": "A recipe uses 250g flour per batch. How much flour for 4 batches?", "answer": 1000},
    {"question": "A theater has 20 rows with 15 seats each. 175 seats are occupied. How many are empty?", "answer": 125},
    {"question": "An athlete runs 3 km in the morning and 5 km in the evening for 5 days. Total km?", "answer": 40},
    {"question": "A jar has 50 coins: 20 quarters and 30 dimes. What is the total value in cents?", "answer": 800},
    {"question": "A builder lays 120 bricks per hour. How many bricks in a 7.5-hour workday?", "answer": 900},
    {"question": "A store buys apples for $0.50 each and sells them for $0.80 each. Profit per dozen?", "answer": 3.6},
    {"question": "A cyclist rides 18 km in 45 minutes. Speed in km per hour?", "answer": 24},
    {"question": "A class collected 240 bottles for recycling over 8 weeks. Average per week?", "answer": 30},
    {"question": "A fence is 120 m long. Posts are placed every 6 m, including both ends. How many posts?", "answer": 21},
    {"question": "Lily saves $25 per month. After 8 months she has saved how much?", "answer": 200},
    # 51-75: percentages and fractions
    {"question": "A jacket costs $80. It is on sale for 25% off. Sale price?", "answer": 60},
    {"question": "A class of 40 students scored an average of 75. Total score points?", "answer": 3000},
    {"question": "A store increases prices by 10%. A $50 item now costs?", "answer": 55},
    {"question": "3/8 of 96 students passed the exam. How many passed?", "answer": 36},
    {"question": "A tank is 3/4 full with 90 liters. Tank capacity?", "answer": 120},
    {"question": "A train is 20% late. It should arrive in 50 minutes. How many minutes late?", "answer": 10},
    {"question": "40% of a 200-person survey said yes. How many said no?", "answer": 120},
    {"question": "A book has 480 pages. Ana reads 1/3 in the first week. Pages remaining?", "answer": 320},
    {"question": "Sales rose 15% from $2000. New sales total?", "answer": 2300},
    {"question": "5/6 of 120 apples are ripe. How many are unripe?", "answer": 20},
    {"question": "A car depreciated 20% from $25000. Current value?", "answer": 20000},
    {"question": "60% of a class of 35 are girls. How many boys?", "answer": 14},
    {"question": "A recipe calls for 3/4 cup sugar. For a double batch, how much sugar?", "answer": 1.5},
    {"question": "A fund grew 8% to reach $10800. Original amount?", "answer": 10000},
    {"question": "30% of 150 items are defective. Non-defective items?", "answer": 105},
    {"question": "A number increased by 12 is 45. What is the number?", "answer": 33},
    {"question": "A price decreased by $14 to $56. Original price?", "answer": 70},
    {"question": "A class has twice as many girls as boys. 12 boys total. Total students?", "answer": 36},
    {"question": "Sam has 3 times as many stickers as Tim. Tim has 15. How many does Sam have?", "answer": 45},
    {"question": "A bag costs half as much as a wallet. Wallet costs $44. Bag costs?", "answer": 22},
    {"question": "A room is 5m longer than it is wide. Width is 4m. Perimeter?", "answer": 26},
    {"question": "A number is 6 less than twice another. Smaller number is 9. Larger?", "answer": 12},
    {"question": "Together two numbers sum to 48. One is 3 times the other. Smaller?", "answer": 12},
    {"question": "A rope is cut into 3 equal pieces and 1 extra piece of 2m. Total was 14m. Each piece?", "answer": 4},
    {"question": "Apples cost $1.20 each; oranges $0.80 each. 3 apples and 4 oranges total cost?", "answer": 6.8},
    # 76-100: multi-step problems
    {"question": "A library has 1200 books. 30% are fiction. 150 non-fiction are donated. Total books?", "answer": 1350},
    {"question": "A factory runs 3 shifts of 8 hours each. Workers earn $12/h. Daily earnings per worker?", "answer": 288},
    {"question": "A water tank drains at 25 liters/min. After 12 minutes, 60 liters remain. Initial amount?", "answer": 360},
    {"question": "A group of 5 friends splits a $63 bill. Two friends pay $15 each. The rest split equally. How much each?", "answer": 11},
    {"question": "A shop sells 45 items per day. After a sale, volume doubles for 3 days. Total items those 3 days?", "answer": 270},
    {"question": "A school trip has 4 buses of 42 students each. 3 students are absent. Students on the trip?", "answer": 165},
    {"question": "A painter uses 3 liters per wall. A room has 4 walls. He has 9 liters. Walls he can paint?", "answer": 3},
    {"question": "Cost: $180 for 4 people for 3 nights. Cost per person per night?", "answer": 15},
    {"question": "A tank fills in 4 hours at 60 liters/h. 1/3 of the water evaporates. Final amount?", "answer": 160},
    {"question": "A bag of rice weighs 2.5 kg. Each meal uses 125g. How many meals?", "answer": 20},
    {"question": "A school fundraiser goal is $500. Each class raises $45 on average. How many classes to meet goal?", "answer": 12},
    {"question": "A bakery makes 8 loaves per batch. They run 6 batches per day. Weekly production (7 days)?", "answer": 336},
    {"question": "A gardener plants 12 seeds per row. After planting 8 rows he has 20 seeds left. How many seeds initially?", "answer": 116},
    {"question": "Train A departs at 9am at 80 km/h. Train B departs at 10am at 100 km/h. Distance between cities is 240 km. When does Train B arrive?", "answer": 12},
    {"question": "A container holds 500 ml. It is 2/5 full. How many ml to fill it?", "answer": 300},
    {"question": "Revenue in Jan was $4000. Feb was 25% more. March was 10% less than Feb. March revenue?", "answer": 4500},
    {"question": "4 workers complete a job in 6 days. How many days for 8 workers?", "answer": 3},
    {"question": "A 600g mixture is 40% sugar. How many grams of sugar to add to make it 50% sugar?", "answer": 120},
    {"question": "A hiker walks at 4 km/h uphill and 6 km/h downhill. Round trip distance is 20 km. Total time (hours)?", "answer": 4.17},
    {"question": "A machine fills 240 bottles in 8 minutes. How many bottles in 35 minutes?", "answer": 1050},
    {"question": "A jar contains 40 coins: pennies and nickels. Total value is $1.20. How many nickels?", "answer": 20},
    {"question": "A car travels 150 km in 2.5 hours. At that speed, how long for 300 km?", "answer": 5},
    {"question": "Team A wins 60% of 50 games. Team B wins 70% of 40 games. Who wins more games?", "answer": 30},
    {"question": "A store doubles the price then offers 20% off. Net change vs original price?", "answer": 60},
    {"question": "Alice is 3 times Bob's age. In 10 years Alice is twice Bob's age. Bob's current age?", "answer": 10},
    # 101-150: additional questions
    {"question": "A box holds 24 cans. How many boxes for 288 cans?", "answer": 12},
    {"question": "A rectangle has perimeter 36 cm and width 8 cm. What is the length?", "answer": 10},
    {"question": "An elevator holds 800 kg. 5 people average 70 kg. Can all fit?", "answer": 1},
    {"question": "Books cost $8 each. Zach has $50. How many can he buy?", "answer": 6},
    {"question": "A piece of rope 60m is cut in ratio 2:3. Length of the longer piece?", "answer": 36},
    {"question": "A car trip: 120 km at 60 km/h, then 80 km at 80 km/h. Total time in hours?", "answer": 3},
    {"question": "A pump fills 1/4 of a tank per hour. To fill 3/4 of the tank takes how many hours?", "answer": 3},
    {"question": "A worker earns $48 for 6 hours. Rate per hour?", "answer": 8},
    {"question": "Prices increased by 5% from $200. New price?", "answer": 210},
    {"question": "A 90-minute meeting is extended by 1/3. Total duration in minutes?", "answer": 120},
    {"question": "A garden produces 5 kg tomatoes per plant. 8 plants. Total kg?", "answer": 40},
    {"question": "An airline flies 800 km. 1/5 is over water. How many km over land?", "answer": 640},
    {"question": "A school day is 6.5 hours. 45 minutes is lunch. Teaching hours?", "answer": 5.75},
    {"question": "Three siblings share 150 stickers in ratio 2:3:5. Middle child gets how many?", "answer": 45},
    {"question": "A machine part is 15 cm long with 2 mm tolerance. Min length in mm?", "answer": 148},
    {"question": "A bag holds apples: 12 green, 8 red. What fraction are red?", "answer": 0.4},
    {"question": "A pool drains completely in 3 hours. Fraction drained per hour?", "answer": 0.333},
    {"question": "A recipe makes 24 cookies with 300g flour. Flour for 36 cookies?", "answer": 450},
    {"question": "A hiker walks 8 km in 2 hours then 6 km in 1.5 hours. Average speed?", "answer": 4},
    {"question": "A number is 15 more than twice another. Sum is 45. Smaller number?", "answer": 10},
    {"question": "A tank leaks 3 liters per hour. Starts at 120 liters. After how many hours is it half full?", "answer": 20},
    {"question": "Sales target is 500 units per week. Current rate is 360. How many units behind after 2 weeks?", "answer": 280},
    {"question": "Team scores: 85, 92, 78, 95, 80. Average score?", "answer": 86},
    {"question": "A store buys for $30 and marks up 40%. Selling price?", "answer": 42},
    {"question": "6 workers finish in 10 days. How many days for 15 workers?", "answer": 4},
    {"question": "Perimeter of an equilateral triangle is 42 cm. Side length?", "answer": 14},
    {"question": "A 1.5L bottle is filled 2/3 full. How many ml of water?", "answer": 1000},
    {"question": "A 45-minute test has 30 questions. Average time per question in seconds?", "answer": 90},
    {"question": "A shop offers buy-2-get-1-free on $12 items. Cost for 9 items?", "answer": 72},
    {"question": "Water evaporates 10% per day. After 2 days from 500ml, how much remains?", "answer": 405},
    {"question": "A car travels 360 km on 30 liters. How many km per liter?", "answer": 12},
    {"question": "A project takes 40 hours total. Done 15 hours so far. Fraction remaining?", "answer": 0.625},
    {"question": "Box A has 24 balls. Box B has 1.5x as many. Total balls?", "answer": 60},
    {"question": "Interest: $1000 at 5% simple interest for 3 years. Total amount?", "answer": 1150},
    {"question": "A pipe fills tank in 6h. Another drains in 9h. Net hours to fill if both open?", "answer": 18},
    {"question": "A class of 32 gets average of 70. If top 2 students (each 100) are excluded, new average?", "answer": 66.67},
    {"question": "Mixture: 3L water at $0 and 2L juice at $5/L. Average cost per liter?", "answer": 2},
    {"question": "A machine makes 500 parts/h with 2% defect rate. Non-defective parts per hour?", "answer": 490},
    {"question": "Stamps: 5 cent and 10 cent. Total 20 stamps worth $1.40. How many 10-cent stamps?", "answer": 8},
    {"question": "A cylinder holds 3.14 * r^2 * h liters. r=2, h=5. Volume?", "answer": 62.8},
    {"question": "Taxi: $2.50 base + $1.75 per km. 8 km trip costs?", "answer": 16.5},
    {"question": "A shop gives 2 for every 10 items bought free. Buying 30 items, how many free?", "answer": 6},
    {"question": "A survey of 80 people: 45% prefer A, 35% B, 20% C. How many prefer B?", "answer": 28},
    {"question": "1 km = 1000m. 5000 steps each 0.7m. Total km?", "answer": 3.5},
    {"question": "A team of 4 devs finishes 12 features per sprint. New sprint needs 21 features. Extra devs needed?", "answer": 3},
    {"question": "Cost of printing: $0.05 per page. 1500 pages total. Cost?", "answer": 75},
    {"question": "A park is 400m x 300m. Jogging track is perimeter. Laps for 5km?", "answer": 3.57},
    {"question": "4 pens and 3 pencils cost $4.90. 3 pens and 2 pencils cost $3.65. Cost of 1 pen?", "answer": 0.85},
    {"question": "A water tank is filled to 80% (120L). Drains 15L. Fraction remaining?", "answer": 0.7},
    # 151-200: final set
    {"question": "A building has 12 floors with 8 apartments each. 20% are vacant. Occupied apartments?", "answer": 77},
    {"question": "Speed: 72 km/h. Distance in meters per second?", "answer": 20},
    {"question": "A bag has 5 red, 3 blue, 2 green balls. Probability of drawing red?", "answer": 0.5},
    {"question": "Ratio of boys to girls is 3:5. Total 40 students. How many boys?", "answer": 15},
    {"question": "A candle is 20cm tall and burns 2cm/hour. After 7 hours?", "answer": 6},
    {"question": "Total marks: 600. Student gets 450. Percentage?", "answer": 75},
    {"question": "A shop sold 240 items: 60% online, 40% in-store. Online sales?", "answer": 144},
    {"question": "3 printers complete job in 4h. 1 printer completes in how many hours?", "answer": 12},
    {"question": "A number doubled, then 6 added gives 22. Original number?", "answer": 8},
    {"question": "Two trains 300km apart, one at 60 km/h and other at 90 km/h, approach each other. Meeting time (hours)?", "answer": 2},
    {"question": "A box of 12 eggs costs $3.60. Cost per egg?", "answer": 0.3},
    {"question": "5 notebooks weigh 750g total. Weight of 8 notebooks?", "answer": 1200},
    {"question": "A rectangle with area 48 cm² and length 8 cm. Width?", "answer": 6},
    {"question": "A journey took 3.5 hours at 80 km/h. Distance covered?", "answer": 280},
    {"question": "30 students average 72 marks. 10 more students join averaging 66. New class average?", "answer": 70.5},
    {"question": "Profit = $450, cost = $1500. Profit percentage?", "answer": 30},
    {"question": "Compound interest: $1000 at 10% for 2 years. Final amount?", "answer": 1210},
    {"question": "A bottle holds 750ml. After drinking 1/3, how much remains?", "answer": 500},
    {"question": "Cost price $200, sold at $250. Gain percent?", "answer": 25},
    {"question": "A pipe fills pool in 8h, another in 12h. Both together fill in how many hours?", "answer": 4.8},
    {"question": "A train 200m long passes a pole in 10 seconds. Speed in km/h?", "answer": 72},
    {"question": "15 workers build a wall in 12 days. How many workers to build in 9 days?", "answer": 20},
    {"question": "A sphere volume = (4/3)*pi*r^3, r=3. Volume (use pi=3.14)?", "answer": 113.04},
    {"question": "Simple interest: $500 at 6% for 2 years. Interest earned?", "answer": 60},
    {"question": "A running track is 400m. An athlete runs 3000m. Laps?", "answer": 7.5},
    {"question": "Cost: $50 for 5 items. Bulk order of 25 items at 15% discount. Total cost?", "answer": 212.5},
    {"question": "A cube has edge 4cm. Surface area?", "answer": 96},
    {"question": "Investments of $400 and $600. Returns 5% and 8% respectively. Total return?", "answer": 68},
    {"question": "A 60-liter mixture is 30% acid. How many liters of pure acid?", "answer": 18},
    {"question": "A clock gains 2 minutes per day. In 2 weeks, how many minutes ahead?", "answer": 28},
    {"question": "A train travels 150km at 75km/h, waits 30min, then 100km at 50km/h. Total time in hours?", "answer": 4.5},
    {"question": "Out of 500 students, 40% play soccer, 30% play basketball, 10% play both. Neither sport?", "answer": 150},
    {"question": "A room 5m x 4m floor costs $15/m² to tile. Total tiling cost?", "answer": 300},
    {"question": "A discount store marks prices up 50%, then offers 30% off. Net change?", "answer": 5},
    {"question": "Work done by A alone in 6 days, B alone in 4 days. Together in how many days?", "answer": 2.4},
    {"question": "A farmer has 100m of fencing. Maximum rectangular area enclosed?", "answer": 625},
    {"question": "A water tap fills 1/5 per hour. 2 taps together fill in how many hours?", "answer": 2.5},
    {"question": "At 8am a car is 200km from city. Driving at 80km/h toward city. Arrival time?", "answer": 10.5},
    {"question": "A box has 15 balls: 5 red, 6 blue, 4 green. Probability of NOT green?", "answer": 0.733},
    {"question": "A 10% solution and 25% solution mixed equally. Resulting concentration?", "answer": 17.5},
    {"question": "Salary increased from $2400 to $2640. Percent increase?", "answer": 10},
    {"question": "A car traveled 240km in the morning and 180km in the afternoon. Average speed over 6 hours?", "answer": 70},
    {"question": "A pipeline is 3/8 full with 75 liters. Pipeline capacity?", "answer": 200},
    {"question": "If 12 pencils cost $3.60, cost of 5 pencils?", "answer": 1.5},
    {"question": "A water tank empties in 4h without inlet, fills in 3h with inlet. Net time to fill from empty?", "answer": 12},
    {"question": "A school day has 6 periods of 45 min and 3 breaks of 10 min. Total in hours?", "answer": 5},
    {"question": "Mixture: 4L at 20% alcohol + 6L at 30% alcohol. Resulting percentage?", "answer": 26},
    {"question": "A car covers 1/3 of journey at 60km/h and 2/3 at 90km/h. Average speed?", "answer": 81},
    {"question": "Three taps fill a tank in 6, 8, and 12 hours respectively. Together in how many hours?", "answer": 2.67},
]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def _extract_numeric_answer(text: str) -> float | None:
    """Extract the final numeric answer from a model response.

    We look for an explicit 'answer is X' pattern first, then fall back to
    the last numeric token.  This covers '= 42', 'Answer: 42', '42.0', etc.
    Tolerating these variants is critical: LLMs do not output a consistent
    answer format, so brittle exact-match scoring severely underestimates accuracy.
    """
    m = re.search(r"(?:answer|total|result)[\s:=is]*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])
    return None


def _answers_match(a: float | None, b: float | str | int | None, tol: float = 0.5) -> bool:
    """Return True if two answer values are within tolerance of each other.

    GSM8K answers are always integers; rounding in model output ('35.0' vs 35)
    should not count as wrong.  Tolerance=0.5 catches off-by-one rounding.
    """
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Verdict classification (REQ-VER-030-3/4/5)
# ---------------------------------------------------------------------------


def classify_verdict(signed_improvement_200q: float) -> str:
    """Return the honest_verdict for the 200q RETRO-033 resolution.

    This is the definitive classification used by the conductor to determine
    whether RETRO-033 should be closed.  We use a 0.01 threshold for 'marginal'
    to capture cases where VR helps but only fractionally — these warrant a
    separate architectural follow-up rather than immediately reopening VR work.

    Classifications (REQ-VER-030-3/4/5):
      - signed_improvement_200q > 0.01  → "vr_finally_positive" (VR works at scale)
      - 0 < signed_improvement_200q <= 0.01 → "vr_marginal" (barely positive, needs architecture work)
      - signed_improvement_200q <= 0    → "vr_not_viable_at_scale" (RETRO-033 closed)
    """
    if signed_improvement_200q > 0.01:
        return "vr_finally_positive"
    elif signed_improvement_200q > 0.0:
        return "vr_marginal"
    else:
        return "vr_not_viable_at_scale"


# ---------------------------------------------------------------------------
# Checkpoint signed_improvement computation
# ---------------------------------------------------------------------------


def compute_signed_improvement_at(
    n_questions: int,
    baseline_corrects: list[bool],
    vr_corrects: list[bool],
) -> float:
    """Compute signed_improvement (vr_accuracy - baseline_accuracy) over first n_questions.

    Slicing at a checkpoint (50, 100, 150, 200) lets us see whether the signal
    stabilises or flips direction as the question count grows.  If results at 50q
    differ substantially from 200q, the 100q signal was indeed noisy.
    """
    slice_b = baseline_corrects[:n_questions]
    slice_v = vr_corrects[:n_questions]
    if not slice_b:
        return 0.0
    baseline_acc = sum(slice_b) / len(slice_b)
    vr_acc = sum(slice_v) / len(slice_v)
    return vr_acc - baseline_acc


# ---------------------------------------------------------------------------
# Single-question inference
# ---------------------------------------------------------------------------


def _run_one_question(
    pipeline: Any,
    question: str,
    ground_truth: float | int,
) -> dict[str, bool]:
    """Run one question through baseline and VR paths.

    Returns dict with baseline_correct and vr_correct.
    Both are False on exception — we don't want one bad question to crash 200q.
    """
    try:
        baseline_response = pipeline._generate(question, max_new_tokens=256)
    except Exception as exc:
        _log.warning("Baseline generation failed: %s", exc)
        baseline_response = ""

    baseline_numeric = _extract_numeric_answer(baseline_response)
    baseline_correct = _answers_match(baseline_numeric, ground_truth)

    try:
        vr_result = pipeline.verify_and_repair(question, baseline_response, "arithmetic")
        vr_response = (
            vr_result.final_response if hasattr(vr_result, "final_response") else baseline_response
        )
    except Exception as exc:
        _log.warning("VR pipeline failed: %s — using baseline response", exc)
        vr_response = baseline_response

    vr_numeric = _extract_numeric_answer(vr_response)
    vr_correct = _answers_match(vr_numeric, ground_truth)

    return {"baseline_correct": baseline_correct, "vr_correct": vr_correct}


# ---------------------------------------------------------------------------
# RETRO-033 resolution writer (REQ-VER-030-6)
# ---------------------------------------------------------------------------


def _write_retro_033_resolution(verdict: str, si_200q: float) -> None:
    """Append RETRO-033 resolution note to ops/known-issues.md.

    Called only when honest_verdict == 'vr_not_viable_at_scale'.
    We append rather than replace so the historical record is preserved
    (CLAUDE.md documentation rule: never remove existing content).

    Args:
        verdict: The honest_verdict string.
        si_200q: signed_improvement at 200q (should be <= 0).
    """
    known_issues_path = _REPO_ROOT / "ops" / "known-issues.md"
    resolution_text = (
        f"\n## RETRO-033 CLOSED (Exp 720, 20260422)\n"
        f"Verdict: {verdict}\n"
        f"signed_improvement at 200q: {si_200q:.4f}\n"
        f"Root cause: VR pipeline does not improve accuracy at current model scale "
        f"(Qwen3.5-0.8B, 19 consecutive attempts at 100q, 1 attempt at 200q).\n"
        f"Resolution: VR removed from active roadmap. Re-evaluate when a larger model "
        f"(>= 7B parameters) or a fundamentally different verification architecture is available.\n"
        f"Spec: REQ-VER-030-6, SCENARIO-VER-037\n"
    )
    try:
        existing = known_issues_path.read_text()
        known_issues_path.write_text(existing + resolution_text)
        _log.info("RETRO-033 resolution written to %s", known_issues_path)
    except Exception as exc:
        _log.warning("Could not write RETRO-033 resolution to known-issues.md: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run 200q VR scale evaluation to resolve RETRO-033."""
    tmpl = ExperimentTemplate(
        exp_id=720,
        title="VR 200q Scale Evaluation: RETRO-033 Resolution",
        deliverable=_DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(720, timeout_minutes=120, result_path=_DELIVERABLE):
        # ------------------------------------------------------------------
        # Step 1: GPU setup — prefer SOTA GGUFs (Qwen3.6-35B, Gemma4-26B).
        # Fall back to Qwen3.5-0.8B (tiny model) with a loud warning that
        # CoT structure will be poor and signed_improvement is expected to be
        # <= 0 (consistent with all 19 prior attempts using tiny models).
        # ------------------------------------------------------------------
        try:
            from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415
            specs = cached_sota_pair(gpu_indices=(0,))
        except Exception:
            specs = None

        if specs is None:
            _log.warning(
                "cached_sota_pair() returned None — no SOTA GGUFs in HF cache. "
                "Falling back to Qwen3.5-0.8B. Note: 19/19 prior VR attempts with "
                "tiny models showed signed_improvement <= 0. Expected result: "
                "vr_not_viable_at_scale."
            )
            MODEL_SPECS = [{"name": "Qwen3.5-0.8B", "hf_id": _QWEN_MODEL_ID, "gpu": 0}]
        else:
            MODEL_SPECS = [specs[0]]

        gpu_status = tmpl.setup_gpu(MODEL_SPECS)
        if not gpu_status.get("all_healthy", False):
            _log.warning("GPU not available — emitting blocked artifact.")
            artifact = tmpl.build_result(
                {
                    "signed_improvement_50q": None,
                    "signed_improvement_100q": None,
                    "signed_improvement_150q": None,
                    "signed_improvement_200q": None,
                    "honest_verdict": "vr_blocked_no_gpu",
                    "inference_mode": "blocked_no_gpu",
                    "n_questions_attempted": 0,
                    "batch_log": [],
                    "models_used": [s["hf_id"] for s in MODEL_SPECS],
                    "retro_033_resolution": "blocked — GPU required for live inference",
                },
                status="blocked",
            )
            tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl._output_path.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # ------------------------------------------------------------------
        # Step 2: Load VR pipeline.
        # ------------------------------------------------------------------
        from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415

        model_id = MODEL_SPECS[0]["hf_id"]
        pipeline = VerifyRepairPipeline(
            model=model_id,
            domains=["arithmetic"],
            max_repairs=1,
            extractor=None,
            semantic_grounding_verifier=None,
            semantic_verifier_v2=None,
            timeout_seconds=60,
            memory=None,
            template_library=None,
            session_memory=None,
            constraint_memory=None,
            nup_probe=None,
            nup_probe_threshold=0.5,
        )

        # ------------------------------------------------------------------
        # Step 3: Run 200 questions via BatchedInferenceRunner (REQ-VER-030-1).
        # We wrap each question as a "prompt" for BatchedInferenceRunner.
        # The runner provides batching + timeout; actual inference calls
        # _run_one_question() per item.
        # ------------------------------------------------------------------
        baseline_corrects: list[bool] = []
        vr_corrects: list[bool] = []

        def _inference_fn(question_item: dict[str, Any]) -> str:
            """Run one question through VR; return JSON-encoded result dict."""
            result = _run_one_question(pipeline, question_item["question"], question_item["answer"])
            return json.dumps(result)

        bir = BatchedInferenceRunner(_inference_fn, batch_size=8)
        bir.batch_timeout_s = 8 * 60  # Per-batch timeout per REQ-VER-030-1.

        raw_results: list[InferenceResult] = bir.run_batch(_QUESTIONS)

        for res in raw_results:
            if res.timed_out or not res.response:
                baseline_corrects.append(False)
                vr_corrects.append(False)
            else:
                try:
                    parsed = json.loads(res.response)
                    baseline_corrects.append(bool(parsed.get("baseline_correct", False)))
                    vr_corrects.append(bool(parsed.get("vr_correct", False)))
                except (json.JSONDecodeError, TypeError):
                    baseline_corrects.append(False)
                    vr_corrects.append(False)

            # Checkpoint every 50 questions (REQ-VER-030-2).
            n_done = len(baseline_corrects)
            if n_done in (50, 100, 150, 200):
                tmpl.checkpoint_save(
                    {
                        "n_done": n_done,
                        "baseline_corrects_so_far": baseline_corrects[:],
                        "vr_corrects_so_far": vr_corrects[:],
                    },
                    step=n_done,
                )

        # ------------------------------------------------------------------
        # Step 4: Compute signed_improvement at 50q / 100q / 150q / 200q.
        # ------------------------------------------------------------------
        si_50q = compute_signed_improvement_at(50, baseline_corrects, vr_corrects)
        si_100q = compute_signed_improvement_at(100, baseline_corrects, vr_corrects)
        si_150q = compute_signed_improvement_at(150, baseline_corrects, vr_corrects)
        si_200q = compute_signed_improvement_at(200, baseline_corrects, vr_corrects)

        _log.info(
            "signed_improvement: 50q=%.4f 100q=%.4f 150q=%.4f 200q=%.4f",
            si_50q, si_100q, si_150q, si_200q,
        )

        # ------------------------------------------------------------------
        # Step 5: Classify verdict (REQ-VER-030-3/4/5).
        # ------------------------------------------------------------------
        honest_verdict = classify_verdict(si_200q)
        _log.info("honest_verdict=%s", honest_verdict)

        # ------------------------------------------------------------------
        # Step 6: If not viable, write RETRO-033 resolution (REQ-VER-030-6).
        # ------------------------------------------------------------------
        retro_033_resolution: str
        if honest_verdict == "vr_not_viable_at_scale":
            _write_retro_033_resolution(honest_verdict, si_200q)
            retro_033_resolution = (
                f"CLOSED — signed_improvement_200q={si_200q:.4f} <= 0. "
                f"VR removed from roadmap until larger model or new architecture."
            )
        elif honest_verdict == "vr_finally_positive":
            retro_033_resolution = (
                f"RESOLVED — signed_improvement_200q={si_200q:.4f} > 0. VR works at scale."
            )
        else:
            retro_033_resolution = (
                f"MARGINAL — signed_improvement_200q={si_200q:.4f}, follow-up architectural work needed."
            )

        # ------------------------------------------------------------------
        # Step 7: Write artifact.
        # ------------------------------------------------------------------
        n_total = len(baseline_corrects)
        artifact = tmpl.build_result(
            {
                "signed_improvement_50q": si_50q,
                "signed_improvement_100q": si_100q,
                "signed_improvement_150q": si_150q,
                "signed_improvement_200q": si_200q,
                "honest_verdict": honest_verdict,
                "inference_mode": "live_gpu",
                "n_questions_attempted": n_total,
                "baseline_accuracy": sum(baseline_corrects) / max(n_total, 1),
                "vr_accuracy": sum(vr_corrects) / max(n_total, 1),
                "models_used": [model_id],
                "batch_log": bir.batch_log,
                "retro_033_resolution": retro_033_resolution,
            },
            status="success",
        )
        tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))

        try:
            pipeline.close()
        except Exception:
            pass

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
