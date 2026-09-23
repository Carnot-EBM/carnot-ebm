"""dc22 level-0 grid-only expert engine (positive control). Constants below were derived from
the public dc22 source (environment_files/dc22/fdcac232/dc22.py) by extract_constants.py."""
C = {'BACKGROUND': 4, 'PADDING': 3, 'BAR_FILL': 3, 'BAR_EMPTY': 0, 'STEP_TOTAL': 128, 'GRID_W': 64, 'GRID_H': 44, 'MOVE': 2, 'SPRITES': [{'name': 'buezna-blrmbx', 'x': 42, 'y': 24, 'layer': 0, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['b', 'buezna', 'sys_click'], 'pixels': [[-1, -1, -1, 9, 9, 9, 9, 9, 9, 9, -1, -1, -1], [-1, -1, -1, 9, 9, 9, 9, 9, 9, 9, -1, -1, -1], [-1, -1, -1, 9, 9, 9, 9, 9, 9, 9, -1, -1, -1], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]]}, {'name': 'buezna-refgps', 'x': 42, 'y': 7, 'layer': 0, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['a', 'buezna', 'sys_click'], 'pixels': [[-1, -1, -1, 8, 8, 8, 8, 8, 8, 8, -1, -1, -1], [-1, -1, -1, 8, 8, 8, 8, 8, 8, 8, -1, -1, -1], [-1, -1, -1, 8, 8, 8, 8, 8, 8, 8, -1, -1, -1], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]}, {'name': 'coorbs-bg', 'x': 32, 'y': 0, 'layer': -7, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': [], 'pixels': [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]}, {'name': 'goknoi', 'x': 24, 'y': 10, 'layer': 0, 'interaction': 'INTANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['goknoi'], 'pixels': [[11, 11], [11, 11]]}, {'name': 'merged-sprite', 'x': 32, 'y': -2, 'layer': 0, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': [], 'pixels': [[-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-1, -1], [-1, -1], [0, 0], [0, 0], [-1, -1], [-1, -1], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2], [0, 0], [0, 0], [-2, -2], [-2, -2]]}, {'name': 'merged-sprite-2', 'x': 8, 'y': 28, 'layer': -2, 'interaction': 'INTANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': [], 'pixels': [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]]}, {'name': 'merged-sprite-2', 'x': 22, 'y': 8, 'layer': -2, 'interaction': 'INTANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': [], 'pixels': [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]]}, {'name': 'plflho1', 'x': 10, 'y': 30, 'layer': 1, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['jfva'], 'pixels': [[14, 14], [14, 14]]}, {'name': 'refgps-plelvb1', 'x': 12, 'y': 14, 'layer': -1, 'interaction': 'INTANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['a', 'omvz', 'orderefgps', 'tovemc'], 'pixels': [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [8, 8, 8, 8, 8, 8, 13, 13, 13, 13], [8, 8, 8, 8, 8, 8, 13, 13, 13, 13], [8, 8, 8, 8, 8, 8, 13, 13, 13, 13], [8, 8, 8, 8, 8, 8, 13, 13, 13, 13]]}, {'name': 'sprite-12', 'x': 41, 'y': 6, 'layer': 0, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['aybe'], 'pixels': [[-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1], [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1], [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1], [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0], [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, {'name': 'sprite-12', 'x': 41, 'y': 23, 'layer': 0, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['aybe'], 'pixels': [[-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1], [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1], [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1], [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0], [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, {'name': 'tacugo-plelvb', 'x': 8, 'y': 20, 'layer': -2, 'interaction': 'INTANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': [], 'pixels': [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]}, {'name': 'tovemc-plelvb1', 'x': 8, 'y': 24, 'layer': 0, 'interaction': 'TANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['b', 'orderefgps', 'tovemc'], 'pixels': [[-2, 9, -2, 9], [9, -2, 9, -2], [-2, 9, -2, 9], [9, -2, 9, -2]]}, {'name': 'tovemc-plelvb2', 'x': 18, 'y': 10, 'layer': -1, 'interaction': 'INTANGIBLE', 'blocking': 'PIXEL_PERFECT', 'tags': ['b', 'omvz', 'orderefgps', 'tovemc'], 'pixels': [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]}, {'name': 'refgps-plelvb2', 'x': 12, 'y': 14, 'layer': -1, 'interaction': 'REMOVED', 'blocking': 'PIXEL_PERFECT', 'tags': ['a', 'omvz', 'orderefgps', 'tovemc'], 'pixels': [[-2, -2, -2, -2, -2, -2, 8, 8, 8, 8], [-2, -2, -2, -2, -2, -2, 8, 8, 8, 8], [-2, -2, -2, -2, -2, -2, 8, 8, 8, 8], [-2, -2, -2, -2, -2, -2, 8, 8, 8, 8], [-2, -2, -2, -2, -2, -2, 8, 8, 8, 8], [-2, -2, -2, -2, -2, -2, 8, 8, 8, 8], [-2, -2, -2, -2, -2, -2, 13, 13, 13, 13], [-2, -2, -2, -2, -2, -2, 13, 13, 13, 13], [-2, -2, -2, -2, -2, -2, 13, 13, 13, 13], [-2, -2, -2, -2, -2, -2, 13, 13, 13, 13]]}, {'name': 'tovemc-plelvb2', 'x': 8, 'y': 24, 'layer': -1, 'interaction': 'REMOVED', 'blocking': 'PIXEL_PERFECT', 'tags': ['b', 'omvz', 'orderefgps', 'tovemc'], 'pixels': [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]}, {'name': 'tovemc-plelvb1', 'x': 18, 'y': 10, 'layer': 0, 'interaction': 'REMOVED', 'blocking': 'PIXEL_PERFECT', 'tags': ['b', 'orderefgps', 'tovemc', 'omvz'], 'pixels': [[-2, 9, -2, 9], [9, -2, 9, -2], [-2, 9, -2, 9], [9, -2, 9, -2]]}], 'VARIANT_COUNTS': {'refgps-plelvb': 2, 'tovemc-plelvb': 2}}

# ---- engine body (appended after the literal constants by build_engine.py) ----------------------
# Grid-only expert engine for dc22 LEVEL 0. Inputs: grid (64x64 frame), action, data. Uses only
# constants derived from the public dc22 source (sprite pixels, positions, layers, rules).
# Hidden state it CANNOT see: the step-counter parity (StepCounter=128 over a 64-pixel bar, so a
# bar pixel covers 2 steps). For a 1-step action it must guess whether the bar advances.
import numpy as np

_Y_OFF = (64 - C["GRID_H"]) // 2  # letterbox offset, scale 1 (grid 64x44)
_PLAYER_COLOR = 14
_BAR_GUESS_ONE_STEP = 1  # pre-registered from VISIBLE rows only: 8 of 15 one-step rows advanced


def _fresh_sprites():
    return [dict(s, pixels=np.asarray(s["pixels"], dtype=np.int64)) for s in C["SPRITES"]]


def _visible(s):
    return s["interaction"] in ("TANGIBLE", "INTANGIBLE")


def _collidable(s):
    return s["interaction"] in ("TANGIBLE", "INVISIBLE")


def _render(sprites, bar):
    view = np.full((C["GRID_H"], C["GRID_W"]), C["BACKGROUND"], dtype=np.int64)
    for s in sorted((s for s in sprites if _visible(s)), key=lambda s: s["layer"]):
        px = s["pixels"]
        h, w = px.shape
        x0, y0 = s["x"], s["y"]
        dx0, dx1 = max(0, x0), min(C["GRID_W"], x0 + w)
        dy0, dy1 = max(0, y0), min(C["GRID_H"], y0 + h)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        reg = px[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0]
        m = reg >= 0
        view[dy0:dy1, dx0:dx1][m] = reg[m]
    out = np.full((64, 64), C["PADDING"], dtype=np.int64)
    out[_Y_OFF : _Y_OFF + C["GRID_H"], 0 : C["GRID_W"]] = view
    bar = max(0, min(64, int(bar)))
    out[63, :] = C["BAR_EMPTY"]
    out[63, :bar] = C["BAR_FILL"]
    return out


def _player(sprites):
    return next(s for s in sprites if "jfva" in s["tags"])


def _configure(sprites, a_state, b_state, aybe_on, pxy):
    """Set the 3 toggle/visibility bits and the player position on a fresh level-0 sprite list."""
    for s in sprites:
        n, xy = s["name"], (s["x"], s["y"])
        if n.startswith("refgps-plelvb"):
            s["interaction"] = "INTANGIBLE" if n[-1] == str(a_state) else "REMOVED"
        elif n.startswith("tovemc-plelvb") and xy == (8, 24):
            if b_state == 0:
                s["interaction"] = "TANGIBLE" if n[-1] == "1" else "REMOVED"
            else:
                s["interaction"] = "INTANGIBLE" if n[-1] == "2" else "REMOVED"
        elif n.startswith("tovemc-plelvb") and xy == (18, 10):
            if b_state == 0:
                s["interaction"] = "INTANGIBLE" if n[-1] == "2" else "REMOVED"
            else:
                s["interaction"] = "INTANGIBLE" if n[-1] == "1" else "REMOVED"
        elif "aybe" in s["tags"] and not aybe_on:
            s["x"] = 500
    p = _player(sprites)
    p["x"], p["y"] = pxy
    return sprites


def _infer(grid):
    """Return (sprites, bar) whose render equals the input grid, else the closest one."""
    g = np.asarray(grid)
    bar = int((g[63] == C["BAR_FILL"]).sum())
    ys, xs = np.nonzero(g[_Y_OFF : _Y_OFF + C["GRID_H"], :] == _PLAYER_COLOR)
    cands = [(int(xs.min()), int(ys.min()))] if len(xs) else [(10, 30)]
    best = None
    for pxy in cands:
        for a in (1, 2):
            for b in (0, 1):
                for ay in (True, False):
                    sp = _configure(_fresh_sprites(), a, b, ay, pxy)
                    d = int((_render(sp, bar) != g).sum())
                    if best is None or d < best[0]:
                        best = (d, sp)
                    if d == 0:
                        return sp, bar
    return best[1], bar


def _hit(sprite, x, y):
    px = sprite["pixels"]
    h, w = px.shape
    if x < sprite["x"] or y < sprite["y"] or x >= sprite["x"] + w or y >= sprite["y"] + h:
        return None
    return int(px[y - sprite["y"], x - sprite["x"]])


def _on_floor(sprites, p):
    for s in sprites:
        if s is p or s["interaction"] != "INTANGIBLE":
            continue
        if any(t in s["tags"] for t in ("ignore", "crzsjq", "vcha")):
            continue
        v = _hit(s, p["x"], p["y"])
        if v is not None and v >= 0:
            return True
    return False


def _collides(sprites, p):
    pp = p["pixels"]
    ph, pw = pp.shape
    for o in sprites:
        if o is p or not _collidable(o) or o["blocking"] == "NOT_BLOCKED":
            continue
        op = o["pixels"]
        oh, ow = op.shape
        x0, x1 = max(p["x"], o["x"]), min(p["x"] + pw, o["x"] + ow)
        y0, y1 = max(p["y"], o["y"]), min(p["y"] + ph, o["y"] + oh)
        if x1 <= x0 or y1 <= y0:
            continue
        a = pp[y0 - p["y"] : y1 - p["y"], x0 - p["x"] : x1 - p["x"]] >= 0
        b = op[y0 - o["y"] : y1 - o["y"], x0 - o["x"] : x1 - o["x"]] >= 0
        if bool(np.any(a & b)):
            return True
    return False


def _next_variant(name):
    if not name[-1].isdigit():
        return ""
    pre, idx = name[:-1], int(name[-1])
    cnt = C["VARIANT_COUNTS"].get(pre, 0)
    if cnt <= 1:
        return ""
    return pre + str(idx % cnt + 1)


def _interaction_for(tags):
    if "omvz" in tags:
        return "INTANGIBLE"
    if "inzejtible" in tags:
        return "INVISIBLE"
    if "buezna" in tags:
        return "INTANGIBLE"
    return "TANGIBLE"


def engine(grid, action, data):
    grid = np.asarray(grid)
    try:
        sprites, bar = _infer(grid)
    except Exception:
        return grid.copy()
    p = _player(sprites)
    action = int(action)
    if action in (1, 2, 3, 4):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
        ox, oy = p["x"], p["y"]
        p["x"], p["y"] = ox + dx * C["MOVE"], oy + dy * C["MOVE"]
        if _collides(sprites, p) or not _on_floor(sprites, p):
            p["x"], p["y"] = ox, oy
        return _render(sprites, bar + _BAR_GUESS_ONE_STEP)
    if action == 6:
        data = data or {}
        gx = int(data.get("x", 0))
        gy = int(data.get("y", 0)) - _Y_OFF
        if not (0 <= gx < C["GRID_W"] and 0 <= gy < C["GRID_H"]):
            return _render(sprites, bar + _BAR_GUESS_ONE_STEP)
        snapshot = [dict(s) for s in sprites]
        button = None
        for s in sorted(sprites, key=lambda s: s["layer"], reverse=True):
            if "ignore" in s["tags"] or "buezna" not in s["tags"] or not _visible(s):
                continue
            v = _hit(s, gx, gy)
            if v is not None and v >= 0:
                button = s
                break
        if button is not None:
            letter = next((t for t in button["tags"] if len(t) == 1), None)
            if letter is not None:
                for s in sprites:
                    if "aybe" in s["tags"]:
                        s["x"] = 500
                group = [
                    s for s in sprites
                    if letter in s["tags"] and s is not button and _visible(s)
                ]
                for s in group:
                    nxt = _next_variant(s["name"])
                    if not nxt:
                        continue
                    live = [o for o in sprites if (o["x"], o["y"], o["name"]) == (s["x"], s["y"], nxt)]
                    pick = next((o for o in live if o["interaction"] != "REMOVED"), live[0] if live else None)
                    if pick is None:
                        continue
                    s["interaction"] = "REMOVED"
                    pick["interaction"] = _interaction_for(pick["tags"])
        if not _on_floor(sprites, p):
            # death animation, then undo to the pre-click snapshot; costs 20 steps = 10 bar pixels
            return _render(snapshot, bar + 10)
        if button is not None:
            return _render(sprites, bar + 1)  # 2 steps always advance exactly one bar pixel
        return _render(sprites, bar + _BAR_GUESS_ONE_STEP)
    return grid.copy()


def is_level_complete(grid):
    return False
