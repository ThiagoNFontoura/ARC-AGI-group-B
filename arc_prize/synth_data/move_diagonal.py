import copy
import random

from arc_prize.synth_data.utils import DatasetInterface, GridPair, generate_grid


class MoveDiagonalDataset(DatasetInterface):
    def _add_random_square(self, grid: list) -> list:
        shape_size = random.randint(1, 3)
        color = random.randint(1, self.num_colors - 1)
        start_row = random.randint(0, self.grid_dim - 1 - shape_size)
        start_col = random.randint(0, self.grid_dim - 1 - shape_size)

        for i in range(shape_size):
            for j in range(shape_size):
                grid[start_row + i][start_col + j] = color

        return grid

    def _shift_grid(
        self, grid: list, vertical_shift: int, horizontal_shift: int
    ) -> list:
        new_grid = copy.deepcopy(grid)

        for _ in range(abs(vertical_shift)):
            if vertical_shift < 0:
                new_grid.pop(0)
                new_grid.append([0] * self.grid_dim)
            elif vertical_shift > 0:
                new_grid.pop()
                new_grid.insert(0, [0] * self.grid_dim)

        for _ in range(abs(horizontal_shift)):
            if horizontal_shift < 0:
                for row in new_grid:
                    row.pop(0)
                    row.append(0)
            elif horizontal_shift > 0:
                for row in new_grid:
                    row.pop()
                    row.insert(0, 0)

        return new_grid

    def generate_pair(self, vertical_shift: int, horizontal_shift: int) -> GridPair:
        input_grid = generate_grid(self.grid_dim, self.grid_dim)
        num_shapes = random.randint(3, 8)
        for _ in range(num_shapes):
            input_grid = self._add_random_square(input_grid)

        output_grid = self._shift_grid(input_grid, vertical_shift, horizontal_shift)

        return GridPair(input=input_grid, output=output_grid)

    def generate_task(self) -> tuple[list[GridPair], GridPair]:
        num_train_pairs = random.randint(1, 4)
        vertical_shift = random.randint(-3, 3)
        horizontal_shift = random.randint(-3, 3)
        train_pairs = [
            self.generate_pair(vertical_shift, horizontal_shift)
            for _ in range(num_train_pairs)
        ]
        test_pair = self.generate_pair(vertical_shift, horizontal_shift)
        return train_pairs, test_pair
