
'''

# initial population of random bitstring
pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
# enumerate generations
for gen in range(n_iter):
# evaluate all candidates in the population
scores = [objective(c) for c in pop]
'''

def generate_random_timetable():
    # Generate a random timetable
    timetable = []
    # Populate timetable with random lectures
    # Add logic to ensure constraints are met
    return timetable

def initial_population(pop_size):
    # Generate an initial population of timetables
    population = []
    for _ in range(pop_size):
        # Create a timetable with random lecture assignments
        timetable = generate_random_timetable()
        population.append(timetable)
    return population


def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # Initialize the population with random bitstrings
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]

    # Keep track of the best solution
    best, best_eval = None, float('inf')

    # Enumerate generations
    for gen in range(n_iter):
        # Evaluate all candidates in the population
        scores = [objective(c) for c in pop]

        # Check for a new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))

        # Select parents using tournament selection
        selected = [selection(pop, scores) for _ in range(n_pop)]

        # Create the next generation
        children = []
        for i in range(0, n_pop, 2):
            # Get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]

            # Crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # Mutation
                mutation(c, r_mut)
                # Store for the next generation
                children.append(c)

        # Replace the population with the new generation
        pop = children

    return best, best_eval


# Tournament selection
def selection(pop, scores, k=3):
    # First random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # Check if better (e.g., perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# Crossover operator
def crossover(p1, p2, r_cross):
    # Children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # Check for recombination
    if rand() < r_cross:
        # Select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # Perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# Mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # Check for a mutation
        if rand() < r_mut:
            # Flip the bit
            bitstring[i] = 1 - bitstring[i]

# Example usage:
# Replace 'objective' with your objective function and adjust other parameters accordingly
# best_solution, best_score = genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut)

        # Check if the selected room is already assigned to a section at this time slot
        if (room, course_code) in room_assignments:
          continue  # If the room is already assigned, redo random choice

        # Check if the room has enough seats for the section
        if room_sizes[room] < int(strength) if strength != 'nan' else 0:
          continue  # If the room doesn't have enough seats, redo random choice

          # Check if the selected instructor is already teaching at this time slot
          if any(instructor in lecture for lecture in timetable):
              continue  # If the instructor is already teaching, redo random choice

          # Check if the selected section is already assigned to a room at this time slot
          if (j, section) in room_assignments:
              continue  # If the section is already assigned, redo random choice

          # If lab, ensure it's scheduled in the last 17 slots
          if status == "lab" and j < col - 17:
              continue  # If lab is not in the last 17 slots, redo random choice

          # If lab, ensure it's scheduled in two consecutive slots
          if status == "lab":
              if lab_count >= 2:
                  continue  # If lab already scheduled in two consecutive slots, redo random choice
              lab_count += 1
          else:
              lab_count = 0  # Reset lab count for regular classes

          # If all constraints are satisfied, remove entry from listTeachers
          '''
          row_indices = np.where(listTeachers == course_info)[0]
          if len(row_indices) > 0:
            # Delete the first occurrence of 'course_info' from the original array
            listTeachers = np.delete(listTeachers, row_indices[0], axis=0)
            print("Array after deleting the row:")
          else:
            print("The value '{}' does not exist in the array.".format(course_info))
          '''
          # If all constraints are satisfied, break out of the while loop
          break
        # Update flag if current lecture is lab

        roomNumbers = ['C301', 'C302', 'C303', 'C304', 'C305', 'C307', 'C308', 'C309', 'C310', 'C311', 'C401', 'C402',
                       'C403', 'C404', 'C405', 'C406', 'C407', 'C408', 'C409', 'C410', 'B130', 'B227', 'B229', 'B230',
                       'A108', 'A211', 'A301', 'A302', 'A303', 'A305', 'A310', 'A311', 'A314', 'A315', 'A316', 'C110',
                       'A118 (MEDC)']
        roomSeats = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 120,
                     60, 120, 60, 60, 60, 60, 60, 60, 60, 120, 60, 120, 120]
        roomSeatsAvailability = np.where(roomSeats == 60, "free", roomSeats)
        roomSeatsAvailability = np.where(roomSeatsAvailability == 120, "free", roomSeatsAvailability)



# Convert values to binary encoding
binary_timetable = []
for i in range (4):
  x = np.unpackbits(population[max_fitness_indices].astype(np.uint8)[:, np.newaxis], axis=1)
  print (x)
  binary_timetable.append(x)

  final_timetable = genetic_algorithm(chosen_population, 0.2)
  print(final_timetable)

  for i in range(chosen_population.shape[0]):
      for j in range(chosen_population.shape[1]):
          for k in range(chosen_population.shape[2]):
              print(i, " ", j, " ", k, "->", chosen_population[i][j][k])
