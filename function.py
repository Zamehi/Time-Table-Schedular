
import random
import numpy as np
import pandas as pd

# ------------------ 1 --------------------
def create_table(listTeachers):
  rows = []
  for i in range(10):
    for j in range(6):
      course_info = random.choice(listTeachers)
      while course_info[0] == '' or course_info[1] == '' or course_info[2] == '':
        course_info = random.choice(listTeachers)
      rows.append(course_info[0] + ' -- ' + course_info[1] + ' -- ' + course_info[2])
  rows = np.array(rows).reshape((10,6))
  return rows


def generate_random_timetable(row,col, listTeachers):
    timetable = []

    # Generate a random timetable
    for i in range (row):
        for j in range (col):
            course_info = random.choice(listTeachers)
            # Select until find an entry
            while course_info[0] == '' or course_info[1] == '' or course_info[2] == '':
                course_info = random.choice(listTeachers)
            # Populate timetable with random lectures
            timetable.append(course_info[0] + ' -- ' + course_info[1] + ' -- ' + course_info[2])
        # converting to 2D array
        timetable= np.array(timetable).reshape((row, col))

    # Populate timetable with random lectures
    # Add logic to ensure constraints are met
    return timetable


### POPULATION
def generate_population(size, listTeachers, row, col):
    #size = 20
    population = []

    for _ in range(size):
        list = generate_random_timetable(listTeachers, 54, 6)
        df = pd.DataFrame(list)
        df.columns = 6
        population.append(df)

    return population



# ---------------- INITIAL POPULATION ------------------
def initial_population(pop_size):
    # Generate an initial population of timetables
    population = []
    for _ in range(pop_size):
        # Create a timetable with random lecture assignments
        timetable = generate_random_timetable()
        population.append(timetable)
    return population
