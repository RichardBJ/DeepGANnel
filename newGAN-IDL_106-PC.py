#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
import time
import statistics
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)
size=800
filterSilence=True

"""
TODO:
Could select only real data with events. Done
Import my save.txt routine to optionally write output.
Rename the To TC with rate constants K13 K12 K23 etc.
Would another state help? I don't think so... 4 states should capture realistic bursting.
Could draw the transition matrix as a graph too? Too fancy. User can do this.
Noise could be more authentic still. 
a. Some slow-wave. freq, phase, amp.
b. Some open channel noise. tf.boolean_mask ...if open add white noise? could be slow.
c. replace noise with Sam noise? was it all TF?
d. SPIKE GENERATION SHOULD BE CHANGED, max should be x not abs(x) and number should be abs(x) not be abs(x) +1. X
e. pink noise should also be tf.function decorated!! X

CHANNEL IS LANE 0
RAW IS LANE 1

RUNNING ON NUMAN-NO_GPU in the TF10 envirnoment which means tf2.10!!! not tf1.0 :-)
"""
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = all logs, '1' = filter out INFO logs, '2' = filter out WARNING logs, '3' = filter out ERROR logs

# Suppress other warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Additional suppression for TensorFlow 2.x
tf.get_logger().setLevel('ERROR')

# List all physical GPUs
gpus = tf.config.list_physical_devices('GPU')

print(f"Number of GPUs detected: {len(gpus)}")


# In[34]:


n = 1  # Number of channels
dt = tf.constant(0.1, dtype=tf.float32)
T = tf.constant(size, dtype=tf.int32)  # In sample points :-)
#Must be a multiple of 2!!!
#Size of channel (relative to the channels so one channel


# In[35]:


@tf.function(experimental_compile=True)
def tf_relaxation(binary_sequence, half_life=4.0, relaxation_amount=0.2):
    """
    Apply exponential relaxation to a 1D binary sequence using vectorized TensorFlow operations.
    
    Args:
    binary_sequence: tf.Tensor, shape [time_steps], sequence of 0s and 1s
    half_life: float, the half-life of the exponential decay
    relaxation_amount: float, the amount of relaxation (positive or negative)
    
    Returns:
    tf.Tensor, shape [time_steps]
    """
    # Convert input to float32
    binary_sequence = tf.cast(binary_sequence, tf.float32)
    
    # Calculate decay rate
    decay_rate = tf.math.log(2.0) / half_life
    
    # Find the indices where steps occur
    steps = tf.not_equal(binary_sequence[1:] - binary_sequence[:-1], 0)
    step_indices = tf.where(steps)[:, 0]
    
    # Calculate the time since each step
    time_steps = tf.range(tf.shape(binary_sequence)[0], dtype=tf.float32)
    time_since_step = time_steps[:, tf.newaxis] - tf.cast(step_indices, tf.float32)
    
    # Calculate the exponential decay for each step
    decay = tf.exp(-decay_rate * tf.maximum(time_since_step, 0.0))
    
    # Calculate the relaxation effect
    step_values = tf.gather(binary_sequence, step_indices + 1) - tf.gather(binary_sequence, step_indices)
    relaxation_effect = relaxation_amount * step_values * decay
    
    # Sum the effects of all steps
    total_relaxation = tf.reduce_sum(relaxation_effect, axis=1)
    
    # Add the relaxation to the original sequence
    relaxed_sequence = binary_sequence + total_relaxation
    
    return relaxed_sequence


# In[36]:


"""
And entirely replaced function, by Claude now that doesn't use tf.probability
it seems. Is it really the same?
This needs to be checked. Certainly much faster this way. About 5x faster... it is getting 10% of the GPU cuda use now. was barely visible before.
"""
@tf.function(experimental_compile=True)
def sim_channel(params):
    kc12, kc21, relaxation, Fnoise, scale, offset, relaxT, kco1, koc2, ko12, ko21 = params
    zero = tf.constant(0.0, dtype=tf.float32)

    # Markov chain simulation
    
    row1 = tf.stack([1-kc12, kc12, zero, zero])
    row2 = tf.stack([kc21, 1-kc21-kco1, kco1, zero])
    row3 = tf.stack([zero, koc2, 1-koc2-ko12, ko12])
    row4 = tf.stack([zero, zero, ko21, 1-ko21])
    
    transition_matrix = tf.stack([row1, row2, row3, row4])
    
    # Initial state distribution
    initial_probs = tf.constant([0.3, 0.3, 0.2, 0.2])
    
    # Manual Markov chain simulation
    tf.random.set_seed(int(time.time() * 1000) % (2**31 - 1))
    def body(i, state, channels):
        next_state_probs = tf.gather(transition_matrix, state)
        next_state = tf.random.categorical(tf.math.log([next_state_probs]), num_samples=1)[0, 0]
        channels = channels.write(i, tf.cast(tf.greater_equal(next_state, 2), tf.float32))
        return i+1, next_state, channels

    initial_state = tf.random.categorical(tf.math.log([initial_probs]), num_samples=1)[0, 0]
    channels = tf.TensorArray(tf.float32, size=T)
    _, _, channels = tf.while_loop(
        lambda i, *_: i < T,
        body,
        (0, initial_state, channels)
    )
    
    channels = channels.stack()
    channels = tf.squeeze(channels)

    # Generate pink noise
    white_noise = tf.random.normal(shape=[T])
    fft_len = T // 2 + 1
    f = tf.range(1, fft_len, dtype=tf.float32)
    spectrum = 1.0 / tf.sqrt(f)
    spectrum = tf.concat([tf.constant([1.0]), spectrum], axis=0)
    white_noise_fft = tf.signal.rfft(white_noise)
    pink_noise_fft = white_noise_fft * tf.cast(spectrum, tf.complex64)
    pink_noise = tf.signal.irfft(pink_noise_fft)
    pink_noise -= tf.reduce_mean(pink_noise)
    pink_noise = pink_noise / tf.math.reduce_std(pink_noise)
    noise = pink_noise * Fnoise

    # Add relaxation
    modified_raw_column = (channels * scale) + offset
    modified_raw_column = tf_relaxation(modified_raw_column, half_life=relaxT, relaxation_amount=relaxation)
    
    modified_raw_column += noise

    # Combine channels and modified raw column
    image = tf.stack([channels, modified_raw_column], axis=1)
    
    # Final safeguard against NaN values
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    
    return image


# In[37]:


# Parameters for the exponential distribution
num_samples = 5
"""kc12, kc21, relaxation, Fnoise, scale, offset, relaxT, kco1, koc2, ko12, ko21"""
kc12 = tf.constant(0.1, dtype=tf.float32)  # Adjust this value as needed
kc21 = tf.constant(0.1, dtype=tf.float32)  # Adjust this value as needed
kco1 = tf.constant(0.01, dtype=tf.float32)
koc2 = tf.constant(0.01, dtype=tf.float32)
ko12 = tf.constant(0.01, dtype=tf.float32)
ko21 = tf.constant(0.01, dtype=tf.float32)


relaxation = tf.constant(0.5, dtype=tf.float32)
Fnoise = tf.constant(.04, dtype=tf.float32)
SCALE = tf.constant(.6, dtype=tf.float32)
#And an offset
OFFSET = tf.constant(-0.4, dtype=tf.float32)
relaxT=25
# nE = tf.constant(200, dtype=tf.int32) #number of events


# Generate training data
training_data = []
lens=[]
"""Actually replace "Anoise" with relaxation later"""
for sample in tqdm(range(num_samples)):   
    params = tf.stack([kc12, kc21, relaxation, Fnoise, SCALE, OFFSET, relaxT, kco1, koc2, ko12,ko21])  # Use tf.stack instead of tf.constant
    segment = sim_channel(params)
    lens.append(sum(abs(segment)))
    training_data.append(segment)
print(f"Average duration was {sum(lens)/len(lens)}")


# In[38]:


#Create REAL Trainging Data
file_path = "/Users/rbj/Documents/GitHub/DeepGANnel/Lina2/4096lina11raw.csv"
df = pd.read_csv(file_path, header=None, names=["Raw", "Channels"])
df = df[["Channels","Raw"]]
# now crop to just one phenotype. There seem multiple in this dataset.
df=df[:75000]
#df=df[:12000]
df = pd.concat([df] * 2, ignore_index=True)
noise = np.random.normal(0, 0.01, df["Raw"].shape)
df["Raw"] += noise
num_rows = (len(df) // size) * size
print(num_rows)
df = df.iloc[:num_rows]
data_array = df.to_numpy()
data_tensor = tf.convert_to_tensor(data_array, dtype=tf.float32)
training_data = tf.reshape(data_tensor, [-1, size, 2])
#Calculate real num_samples!
num_samples= tf.shape(training_data)[0]
#num_samples = 10 #debug
#Only use windows where something happened!
filterSilence = True
if filterSilence:
    first_column = training_data[:, :, 0]
    all_same = tf.reduce_all(tf.equal(first_column, first_column[:, 0:1]), axis=1)
    
    # Filter out batches where all values in the first column are the same
    training_data = tf.boolean_mask(training_data, ~all_same)



# In[39]:


df["Raw"].plot()


# In[40]:


def plotter(data):
    # Create a figure with two subplots (panels)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
     # Flatten the axs array for easy iteration
    axs = axs.flatten()
    
    for i in range(4):
        axs[i].plot(data[i])
        #axs[i].set_ylim([-200, 200])
    plt.tight_layout()
    plt.show()

plotter(training_data)


# In[ ]:





# In[41]:


# Define the generator model
"""kc12, kc21, relaxation, Fnoise, SCALE, OFFSET, relaxT, kco1, koc2, ko12, ko21"""
gen_input_len=11

def make_generator_model():
    noise_input = tf.keras.layers.Input(shape=(gen_input_len,))
    x = tf.keras.layers.Dense(128)(noise_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Output layer without activation
    raw_output = tf.keras.layers.Dense(gen_input_len, activation='softplus')(x)
    
    # Apply appropriate activations/scaling to each output
    kc12 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 1e-6, 1e-6, 1.0))(raw_output[:, 0:1])
    kc21 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 1e-6, 1e-6, 1.0))(raw_output[:, 1:2])
    relaxation = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0))(raw_output[:, 2:3])
    Fnoise = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x), 0.0, 1.0))(raw_output[:, 3:4])
    scale = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 0.1, 0.1, 10.0))(raw_output[:, 4:5])
    offset = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0))(raw_output[:, 5:6])
    relaxT = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x * 10.0, 1.0, size - 1.0))(raw_output[:, 6:7])
    kco1 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 1e-3, 1e-6, 1.0))(raw_output[:, 7:8])
    koc2 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 1e-6, 1e-6, 1.0))(raw_output[:, 8:9])
    ko12 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 1e-6, 1e-6, 1.0))(raw_output[:, 9:10])
    ko21 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.abs(x) + 1e-6, 1e-6, 1.0))(raw_output[:, 10:11])
    
    output = tf.keras.layers.Concatenate()([kc12, kc21, relaxation, Fnoise, scale, offset, relaxT, kco1, koc2, ko12, ko21])
    
    return tf.keras.Model(inputs=noise_input, outputs=output)

# Define the discriminator model batch, record len, channels = events then noise
num_points = T.numpy().item()
def make_discriminator_model():
    input_shape = (size,2) 
    inputs = tf.keras.Input(shape=input_shape)

    # Reshape input to add channel dimension
    x = tf.keras.layers.Reshape((size, 2))(inputs)
    
    # 1D Convolutional layers
    x = tf.keras.layers.Conv1D(8, kernel_size=5, strides=2, padding='same', activation='leaky_relu')(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=5, strides=2, padding='same', activation='leaky_relu')(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding='same', activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3) (x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    """
    # Dense layers
    x = tf.keras.layers.Dense(256, activation='leaky_relu')(x)
    x = tf.keras.layers.Dense(128, activation='leaky_relu')(x)"""
    
    # Output layer
    outputs = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True))

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)


# In[51]:


# Training step
@tf.function(experimental_compile=True)
def train_step(real_data):
    """unconventionally we set a random seed here before running the model. Usually the randomness
    is a feature of a generator model so it learns to make a feasible image from any starting point.
    Here there is likely 1 set of 'perfect' parameters, but the difference trace-to-trace is delivered by
    the stochastic nature of sim_channel itself.  That said there is concern this may make the generator
    very sensitive to implimentation at a later dates?"""
    tf.random.set_seed(123)
    noise = tf.random.normal([batch_size, gen_input_len])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_params = generator(noise, training=True)

        try:
            generated_data = tf.map_fn(
                sim_channel, 
                generated_params, 
                fn_output_signature=tf.float32,
                parallel_iterations=1  # This can help with TensorArray issues
            )
            generated_data = tf.ensure_shape(generated_data, [batch_size, size, 2])
        except Exception as e:
            tf.print("sim_channel error:", e)
            return tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
        
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Training loop
@tf.function(experimental_compile=True)
def train(dataset, epochs):
    steps_per_epoch = math.floor(num_samples / batch_size)
    #steps_per_epoch =2
    tf.print("steps per epoch", steps_per_epoch)
    for epoch in range(getREALepoch(),epochs,1):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Initialize loss accumulators for each epoch
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        
        for step, batch in tqdm(enumerate(dataset), total=steps_per_epoch, ncols=60):
            #clear_output(wait=True)
            if step >= steps_per_epoch:
                break  # Move to the next epoch          
            try:
                gen_loss, disc_loss = train_step(batch)
                #tf.print("gen_loss",gen_loss)
                epoch_gen_loss += gen_loss
                #tf.print("epoch_gen_loss",gen_loss)
                epoch_disc_loss += disc_loss
            except Exception as e:
                print(f"Error during training: {e}")
                break

        #tf.print("epoch_gen_loss",epoch_gen_loss)
        # Calculate average losses for the epoch
        avg_gen_loss = epoch_gen_loss / steps_per_epoch
        avg_disc_loss = epoch_disc_loss / steps_per_epoch
        clear_output(wait=True)
        tf.print(f"Epoch {epoch + 1}/{epochs} - "
              f"Generator Loss: {avg_gen_loss:.8f}, "
              f"Discriminator Loss: {avg_disc_loss:.8f}")
        checkpoint.save(file_prefix = 'markovCheckpoints/checkpoint')
        # Generate and plot sine waves
        egs=2
        
        # Generate and plot data
        EgNoise = tf.random.normal([egs, gen_input_len])
        #print("EgNoise", EgNoise[0])
        generated_params = generator(EgNoise, training=False)
        #print(steps_per_epoch)
        # Define parameter names kc12, kc21, spikeMax, Fnoise, scale, offset, nSpikes, kco1, koc2, ko12, ko21
        param_names = ['kc12', 'kc21', 'relaxation', 'Fnoise', 'scale',"offset","relaxT", "kco1", "koc2","ko12","ko21"]
        random_index1 = tf.random.uniform(shape=[], minval=0, maxval=egs-1, dtype=tf.int32)

        params_list = generated_params[random_index1].numpy().tolist()
        #Might be fun to collect these up to plot convergence if wanted?
                
        # Print each parameter with its name
        with tf.io.gfile.GFile('output.csv', mode='a') as file:
            # Check if the file is empty to write the header
            if file.tell() == 0:
                file.write(','.join(param_names) + '\n')

            for name, param in zip(param_names, params_list):
                tf.print(f"{name}: {round(param, 2)}|", end = " ")
                file.write(f"{name},{round(param, 2)}\n")
         
        gen_waves=[]
        for i in range(egs):
            gen_waves.append( sim_channel(generated_params[i]) )

        """
        # Create a figure with two subplots (panels)
        fig, axs = plt.subplots(egs, 1, figsize=(10, 6))
        for i in range(egs):
            axs[i].plot(gen_waves[i] )
            #axs[i].set_ylim([0,1])
        plt.tight_layout()
        plt.show()
        """
        #tf.print(tf.shape(gen_waves))
        random_index2 = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(training_data)[0], dtype=tf.int32)

        biPlotter([gen_waves[random_index1],training_data[random_index2]], random_index1, random_index2, epoch)
              
        if (epoch + 1) % 10 == 0:
            print(f"Completed {epoch + 1} epochs")
        if writeNow:
            writeMe(epoch=epoch)
            


# In[52]:


def biPlotter(data, n, m, epoch):
    # Create a figure with two subplots (panels)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(data[0] )
    axs[0].set_title(f"Generated Wave, record {n}, epoch {epoch}")
    axs[0].set_ylim([-1,1.2])
    axs[1].plot(data[1] )
    axs[1].set_title(f"Training Data, record {m}")
    axs[1].set_ylim([-1,1.2])
    plt.tight_layout()
    plt.savefig(f"chanFigs/fig{epoch}.png")
    plt.show()
    
    


# In[53]:


def writeMe(samples=100, dt=0.1, epoch=0, file="markovData/output.parquet"):
    sampleNoise = tf.random.normal([samples, gen_input_len])
    generated_params = generator(sampleNoise, training=False)
    gen_waves=[]
    for i in range(samples):
        gen_waves.extend( sim_channel(generated_params[i]) )
    df = pd.DataFrame(gen_waves, columns=["Channels", "Noisy Current"])
    df["Time"] = dt * pd.Series(range(len(df)))
    df = df[["Time", "Channels", "Noisy Current"]]
    df.to_parquet(f"{epoch}_{file}")
    print(f"Data saved to {epoch}_{file}")
    


# In[54]:


def getREALepoch() -> int:
    import glob
    import os
    import re
    """
    Save the current image to the working directory of the program.
    """
    currentfiles = glob.glob("markovCheckpoints/*.index")
   
    numList = [0]
    for file in currentfiles:
        i = os.path.splitext(file)[0]
        try:
            pattern = r'-(\d+)'
            num = re.findall(pattern, i)[0]
            numList.append(int(num))
        except IndexError:
            pass
    numList = sorted(numList)
    return numList[-1]


# In[55]:


batch_size = 10
training_dataset = tf.data.Dataset.from_tensor_slices(
    training_data).shuffle(5000).batch(batch_size, drop_remainder=True).repeat()


# In[56]:


learning_rate_value = 1e-6
# Convert the learning rate value to the appropriate dtype
generator_optimizer.learning_rate.assign(tf.cast(learning_rate_value, generator_optimizer.learning_rate.dtype))

learning_rate_value = 1e-5
discriminator_optimizer.learning_rate.assign(tf.cast(learning_rate_value, discriminator_optimizer.learning_rate.dtype))


# In[57]:


@tf.function(experimental_compile=True)
def read_set_lr():
    with open("lr.txt", "r+") as my_file:
        data = my_file.read()
        split = data.split('\n')
        parse_lr_from_file = lambda string: float(string.split(":")[1])
        new_gen_lr = parse_lr_from_file(split[1])
        new_disc_lr = parse_lr_from_file(split[2])
        generator_optimizer.learning_rate.assign(tf.cast(new_gen_lr, generator_optimizer.learning_rate.dtype))
        discriminator_optimizer.learning_rate.assign(tf.cast(new_disc_lr, discriminator_optimizer.learning_rate.dtype))
        tf.print(f"dLR: {discriminator_optimizer.learning_rate.numpy():.3e},\
                    gLR: {generator_optimizer.learning_rate.numpy():.3e}")
read_set_lr()


# In[58]:


checkpoint_dir = "markovCheckpoints/checkpoints"
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)


# In[ ]:


writeNow=False
epochs=20000
train(training_dataset, epochs)


# In[ ]:


tf.keras.backend.set_value(generator_optimizer.learning_rate,1e-5)
tf.keras.backend.set_value(discriminator_optimizer.learning_rate,1e-4)


# In[ ]:


checkpoint.restore('markovCheckpoints/checkpoint-3').assert_existing_objects_matched()

tf.keras.backend.set_value(generator_optimizer.learning_rate,1e-5)
tf.keras.backend.set_value(discriminator_optimizer.learning_rate,1e-4)
"""gen_loss_list=[]
disc_loss_list=[]"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




