import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    #Static Variables
    g = 9.81 #m/s2
    airD = 1.29 #kg/m3
    oilD = 860 #kg/m3
    platesD = 0.009 #m
    airV = 1.83*np.power(10.0,-5) #pas
    U = 4000 #V
    mHeightSpray = .1 #m 
    n = int(input("Number of simulations to run: "))
    chargesList = np.empty(n)

    e = 1.602e-19


    #Visual Stuff
    time_list     = []
    position_list = []
    velocity_list = []


    for i in range (0,n):

        #Gauss Bell for generation of radii (maybe a bit overcomplicated, we know 😁)
        mu = math.log(math.sqrt(10**-6 * 10**-5)) #ln(hypotenuse bwetween our orders of magnitud), approximately -12.664
        sigma = 0.4 #Standard Deviation
        while True:
            r = np.random.lognormal(mu, sigma)
            #print(r)
            if 0.000001 <= r <= 0.00001: 
                break

        vDrop = (4/3) * np.pi * (r**3)
        mDrop = vDrop * oilD #Drop's mass
        fgDrop = mDrop * g #Gravitational Force ejerced by the Oil Drop
        bfDrop = vDrop * airD * g #Bouyancy Force
        electricCamp = U / platesD         # Electric field

        
        k = np.random.randint(1, 10)  #Calculates a random multiple of e between 1 and 9
        q_true = k * e  # Assigns the multiple

        #Simulates until terminal velocity
        current_velocity = 0.0
        previous_velocity = 0.0
        current_height = platesD / 2
        t = 0.0
        dt = 1e-5
        max_time = 2.0

        while t < max_time:
             #Forces
            electricForce = q_true * electricCamp
            drag_force    = 6 * np.pi * airV * r * current_velocity
            net_force     = fgDrop - bfDrop + electricForce - np.sign(current_velocity)*abs(drag_force)
            acceleration  = net_force / mDrop

            #Update velocity, position, time
            current_velocity += acceleration * dt
            current_height   += current_velocity * dt
            t                += dt

            #Visual stuff of first particle
            if i == 0:
                time_list.append(t)
                position_list.append(current_height)
                velocity_list.append(current_velocity)

            #Criteria if the acceleration is minimal, then we are at terminal velocity.
            if abs(acceleration) < 1e-6 and abs(current_velocity-previous_velocity) < 1e-6:
                break
            previous_velocity = current_velocity

        v_t = current_velocity  # Terminal velocity magnitude

        # Estimate the charge by inverting the dynamic equilibrium:
        #    0 = Fg − Fb + q E − 6π η r v_t
        #    → q_est = [6π η r v_t − (Fg − Fb)] / E
        drag_term = 6 * np.pi * airV * r * v_t
        q_est = (drag_term - (fgDrop - bfDrop)) / electricCamp

        # Saves the estimated charge
        chargesList[i] = q_est
        print(f"Sim {i}: q_true = {q_true:.2e}, q_est = {q_est:.2e}")
        print(i)



    nonzero_charges = chargesList[chargesList != 0] #If drops are generated with charge 0
    sorted_charges = np.sort(nonzero_charges) #Lower to higher
    differences = np.abs(np.diff(sorted_charges)) #Difference between consecutives charges

    differences = differences[differences > 1e-19] #Gets rid of non-useful data

    e_estimated = np.min(differences) #The difference will be in multiples of e (1e, 2e, 3e, 4e), so we just want the e difference, which is the lowest

    qdropcharges = np.mean(nonzero_charges)
    print("Average charge of droplets without zeros: ", qdropcharges, " C") #A bit useless, but useful for debug

    print("Estimated charge of e: ", e_estimated)



    #Visual Stuff
    time_array     = np.array(time_list)
    position_array = np.array(position_list)
    velocity_array = np.array(velocity_list) 

    #Calculate relative displacement starting from zero:
    rel_disp    = position_array - position_array[0]
    #Convert to micrometers for better visibility
    rel_disp_um = rel_disp * 1e6  #1 m = 1e6 micrometers

    #Plot displacement vs time
    plt.figure()
    plt.plot(time_array, rel_disp_um, label='Drop Displacement')  #Plot the displacement
    plt.title('Displacement vs Time')  #Title for plot
    plt.xlabel('Time (s)')  #Label for x-axis
    plt.ylabel('Displacement (μm)')  #Label for y-axis
    plt.grid(True)  #Add grid to plot
    plt.legend()  #Add legend to plot

    #Plot velocity vs time
    plt.figure()
    plt.plot(time_array, velocity_array)  #Plot the velocity
    plt.title('Velocity vs Time (fall)')  #Title for the plot
    plt.xlabel('Time (s)')  # Label for x-axis
    plt.ylabel('Velocity (m/s)')  # Label for y-axis
    plt.grid(True)  # Add grid to plot

    plt.show()  # Display the plots


main()