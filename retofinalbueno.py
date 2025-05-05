import numpy as np
import matplotlib.pyplot as plt

# Variables estáticas
g = 9.81  # m/s^2
airD = 1.29  # kg/m^3
oilD = 860  # kg/m^3
platesD = 0.001  # m
airV = 1.83e-5  # Pa·s
U = 4000  # V
mHeightSpray = 0.1  # m

a_placas = 0.049
resistencia = 1e9
voltaje_Fuente = 280
epsilon_0 = 8.854e-12

e = 1.602e-19

# Número de simulaciones
n = int(input("Number of simulations to run: "))
chargesList = np.zeros(n)

# Variables para visualización
time_list = []
position_list = []
velocity_list = []

for i in range(n):
    # Generar radio aleatorio lognormal entre 1e-6 y 1e-5
    mu = np.log(np.sqrt(1e-6 * 1e-5))
    sigma = 0.4
    while True:
        r = np.random.lognormal(mu, sigma)
        if 1e-6 <= r <= 1e-5:
            break

    vDrop = (4/3) * np.pi * r**3
    mDrop = vDrop * oilD
    fgDrop = mDrop * g
    bfDrop = vDrop * airD * g
    electricCamp = U / platesD

    k = np.random.randint(1, 10)
    q_true = k * e

    current_velocity = 0.0
    previous_velocity = 0.0
    current_height = platesD / 2
    t = 0.0
    dt = 1e-5
    max_time = 2.0

    while t < max_time:
        electricForce = q_true * electricCamp
        drag_force = 6 * np.pi * airV * r * current_velocity
        net_force = fgDrop - bfDrop + electricForce - np.sign(current_velocity) * abs(drag_force)
        acceleration = net_force / mDrop

        current_velocity += acceleration * dt
        current_height += current_velocity * dt
        t += dt

        if i == 0:
            time_list.append(t)
            position_list.append(current_height)
            velocity_list.append(current_velocity)

        if abs(acceleration) < 1e-6 and abs(current_velocity - previous_velocity) < 1e-6:
            break

        previous_velocity = current_velocity

    v_t = current_velocity
    drag_term = 6 * np.pi * airV * r * v_t
    q_est = (drag_term - (fgDrop - bfDrop)) / electricCamp
    chargesList[i] = q_est

    print(f"Sim {i+1}: q_true = {q_true:.6e}, q_est = {q_est:.6e}")

nonzero_charges = chargesList[chargesList != 0]
sorted_charges = np.sort(nonzero_charges)
differences = np.abs(np.diff(sorted_charges))
differences = differences[differences > 1e-19]
e_estimated = np.min(differences) if len(differences) > 0 else 0

qdropcharges = np.mean(nonzero_charges)
print(f"Average charge of droplets without zeros: {qdropcharges:.6e} C")
print(f"Estimated charge of e: {e_estimated:.6e}")

# Visualización
rel_disp = np.array(position_list) - position_list[0]
rel_disp_um = rel_disp * 1e6

plt.figure()
plt.plot(time_list, rel_disp_um)
plt.title('Displacement vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (μm)')
plt.grid(True)

plt.figure()
plt.plot(time_list, velocity_list)
plt.title('Velocity vs Time (fall)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)

# Parte de circuito RC
capacitancia = a_placas * epsilon_0 / platesD
c_tiempo = resistencia * capacitancia
tiempo = np.linspace(0, 2, 1000)
volt_capacitor = voltaje_Fuente * (1 - np.exp(-tiempo / c_tiempo))
c_electrico_cap = volt_capacitor / platesD
z = np.linspace(0, platesD, 100)
d_potencial = np.outer(volt_capacitor, z / platesD)

plt.figure()
plt.plot(tiempo, volt_capacitor)
plt.title('Voltaje del capacitor vs. tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)

plt.figure()
plt.plot(tiempo, c_electrico_cap)
plt.title('Campo eléctrico vs. tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Campo eléctrico (V/m)')
plt.grid(True)

plt.figure()
plt.imshow(d_potencial, aspect='auto', extent=[z[0]*1e6, z[-1]*1e6, tiempo[-1], tiempo[0]])
plt.colorbar(label='Potencial (V)')
plt.xlabel('Posición entre placas (μm)')
plt.ylabel('Tiempo (s)')
plt.title('Distribución de potencial entre placas')
plt.gca().invert_yaxis()

plt.show()