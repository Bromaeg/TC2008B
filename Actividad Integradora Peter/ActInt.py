
# Actividad Integradora
# Miguel Angel Tena Garcia
# A01709653

#Librerias necesarias para correr el programa

#pip install mesa
#pip install matplotlib
#pip install numpy
#pip install enum
#pip install IPython


# Importar las librerías necesarias
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from enum import Enum
from IPython.display import HTML

# Definir los tipos de celda
class CellType(Enum):
    EMPTY = 0
    FISH = 1
    SHARK = 2

# Definicion del agente de pez
class Fish(Agent):
    def __init__(self, unique_id, model, energy):
        # Inicializar el agente, contador de fertilidad igual a 0 y energía igual a la energía inicial
        super().__init__(unique_id, model)
        self.energy = energy
        self.fertility_counter = 0

    def step(self):
        #Step del agente, moverse, perder una unidad de energia, reproducirse.
        self.move()
        self.energy -= 1
        self.reproduce()
        # Los peces pierden una unidad de energía en cada cronón de tiempo.
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move(self):
        # Moverse a una celda vacía
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        empty_steps = [step for step in possible_steps if self.model.grid.is_cell_empty(step)]
        if empty_steps:
            new_position = self.random.choice(empty_steps)
            self.model.grid.move_agent(self, new_position)

    def reproduce(self):
        # Incrementar el contador de fertilidad y reproducirse si el contador es mayor o igual al umbral de fertilidad
        self.fertility_counter += 1
        if self.fertility_counter >= self.model.fish_fertility_threshold:
            offspring = Fish(self.model.next_id(), self.model, self.model.fish_initial_energy)
            self.model.grid.place_agent(offspring, self.pos)
            self.model.schedule.add(offspring)
            self.fertility_counter = 0

# Definicion del agente de los tiburones
class Shark(Agent):
    def __init__(self, unique_id, model, energy):
        super().__init__(unique_id, model)
        self.energy = energy
        self.fertility_counter = 0

    def step(self):
        #Step del agente, moverse, comer, perder una unidad de energia, reproducirse.
        self.move_and_eat()
        self.reproduce()
        self.energy -= 1
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move_and_eat(self):
        # Moverse a una celda con pez y comerlo, si no hay peces moverse a una celda vacía
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)  # Cambiar a moore=True para permitir un movimiento diagonal
        fish_positions = [step for step in possible_steps if
                          self.model.grid.get_cell_list_contents([step]) and
                          isinstance(self.model.grid.get_cell_list_contents([step])[0], Fish)]
        if fish_positions:
            new_position = self.random.choice(fish_positions)
            fish = self.model.grid.get_cell_list_contents([new_position])[0]
            self.model.grid.remove_agent(fish)
            self.model.schedule.remove(fish)
            self.model.grid.move_agent(self, new_position)
            self.energy += self.model.shark_energy_from_fish
        else:
            empty_steps = [step for step in possible_steps if self.model.grid.is_cell_empty(step)]
            if empty_steps:
                new_position = self.random.choice(empty_steps)
                self.model.grid.move_agent(self, new_position)


    def reproduce(self):
        # Incrementar el contador de fertilidad y reproducirse si el contador es mayor o igual al umbral de fertilidad
        self.fertility_counter += 1
        if self.fertility_counter >= self.model.shark_fertility_threshold:
            offspring = Shark(self.model.next_id(), self.model, self.model.shark_initial_energy)
            self.model.grid.place_agent(offspring, self.pos)
            self.model.schedule.add(offspring)
            self.fertility_counter = 0

# Añadir un metodo para recolectar la informacion del grid.
def compute_grid(model):
    return model.get_grid()

# DEfinicion del modelo Wa-Tor.
class WaTorModel(Model):
    # Definir el numero de agentes, el grid, el schedule y el datacollector
    def find_empty_cell(self):
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        while not self.grid.is_cell_empty((x, y)):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
        return x, y
    
    def __init__(self, width, height, N_fish, N_sharks, fish_initial_energy, shark_initial_energy,
                 fish_fertility_threshold, shark_fertility_threshold, shark_energy_from_fish):
        self.num_agents = N_fish + N_sharks
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Grid": compute_grid})
        self.fish_initial_energy = fish_initial_energy
        self.shark_initial_energy = shark_initial_energy
        self.fish_fertility_threshold = fish_fertility_threshold
        self.shark_fertility_threshold = shark_fertility_threshold
        self.shark_energy_from_fish = shark_energy_from_fish

        # Inicializar current_id a 0
        self.current_id = 0

        # Crear Peces
        for _ in range(N_fish):
            x, y = self.find_empty_cell()
            fish = Fish(self.next_id(), self, fish_initial_energy)
            self.grid.place_agent(fish, (x, y))
            self.schedule.add(fish)

        # Crear Tiburones
        for _ in range(N_sharks):
            x, y = self.find_empty_cell()
            shark = Shark(self.next_id(), self, shark_initial_energy)
            self.grid.place_agent(shark, (x, y))
            self.schedule.add(shark)


    def step(self):
          # Recolectar datos cada 10 iteraciones
        self.datacollector.collect(self)
        self.schedule.step()


    def get_grid(self):
        grid = np.zeros((self.grid.width, self.grid.height))
        for cell in self.grid.coord_iter():
            content, (x, y) = cell
            if content:
                if isinstance(content[0], Shark):
                    grid[x][y] = CellType.SHARK.value
                elif isinstance(content[0], Fish):
                    grid[x][y] = CellType.FISH.value
        return grid

# Parametros para la simulacion
width = 60
height = 85
N_fish = 140
N_sharks = 45
fish_initial_energy = 20
shark_initial_energy = 4
fish_fertility_threshold = 6
shark_fertility_threshold = 10
shark_energy_from_fish = 8

MAX_ITERATIONS = 100

# Crear una instancia del modelo de Wa-Tor y correr la simulacion
model = WaTorModel(width, height, N_fish, N_sharks, fish_initial_energy, shark_initial_energy,
                   fish_fertility_threshold, shark_fertility_threshold, shark_energy_from_fish)

# Correr la simulacion
for i in range(MAX_ITERATIONS):
    model.step()

# Graficar el grid
all_grid = model.datacollector.get_model_vars_dataframe()


fig, axis = plt.subplots(figsize=(10,6))
axis.set_xticks([])
axis.set_yticks([])
patch = plt.imshow(all_grid.iloc[0]["Grid"], cmap='viridis')

# Funcion para animar el grid
def animate(i):
    patch.set_data(all_grid.iloc[i]["Grid"])

# Animar el grid
anim_html = animation.FuncAnimation(fig, animate, frames=len(all_grid)).to_jshtml()
HTML(anim_html)


#__END__ 
