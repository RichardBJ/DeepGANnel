import numpy as np
import tensorflow as tf
import pymc3 as pm
import theano.tensor as tt
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define constants
SIZE = 200
NUM_SAMPLES = 1000
EPOCHS = 100

# Simplified channel simulation function
def sim_channel(params):
    To, Tc, Anoise, Fnoise, scale, offset = params
    t = np.arange(SIZE)
    channel = np.random.choice([0, 1], size=SIZE, p=[1-To, To])
    noise = np.random.normal(0, Anoise, SIZE) + np.sin(2 * np.pi * Fnoise * t)
    return np.stack([channel, channel * scale + offset + noise], axis=-1)

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=5, strides=2, padding='same', activation='leaky_relu', input_shape=(SIZE, 2)),
        tf.keras.layers.Conv1D(128, kernel_size=5, strides=2, padding='same', activation='leaky_relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model

# Loss function for the discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Create and compile the discriminator
discriminator = make_discriminator_model()
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Generate some fake "real" data for demonstration
real_data = np.array([sim_channel([0.1, 0.1, 0.01, 0.01, 0.25, -0.1]) for _ in range(NUM_SAMPLES)])

# Define the Bayesian model
with pm.Model() as model:
    # Priors for parameters
    To = pm.Beta('To', alpha=2, beta=5)
    Tc = pm.Beta('Tc', alpha=2, beta=5)
    Anoise = pm.HalfNormal('Anoise', sigma=0.1)
    Fnoise = pm.HalfNormal('Fnoise', sigma=0.1)
    scale = pm.HalfNormal('scale', sigma=0.5)
    offset = pm.Normal('offset', mu=0, sigma=0.1)
    
    # Custom Theano Op for sim_channel
    class SimChannelOp(tt.Op):
        itypes = [tt.dscalar] * 6
        otypes = [tt.dmatrix]
        
        def perform(self, node, inputs, outputs):
            outputs[0][0] = sim_channel(inputs)
    
    sim_channel_op = SimChannelOp()
    
    # Generate synthetic data
    synthetic_data = sim_channel_op(To, Tc, Anoise, Fnoise, scale, offset)
    
    # Placeholder for discriminator feedback (will be updated during training)
    discriminator_output = pm.Normal('discriminator_output', mu=0, sigma=1)
    
    # Likelihood (we want the discriminator to output close to 1 for generated data)
    pm.Normal('likelihood', mu=discriminator_output, sigma=0.1, observed=1)

# Training loop
for epoch in tqdm(range(EPOCHS)):
    # Sample from the posterior
    with model:
        trace = pm.sample(500, tune=500, chains=2, cores=1)
    
    # Generate synthetic data using the samples
    synthetic_data = []
    for i in range(NUM_SAMPLES):
        idx = np.random.randint(len(trace))
        params = [trace.get_values('To')[idx], trace.get_values('Tc')[idx],
                  trace.get_values('Anoise')[idx], trace.get_values('Fnoise')[idx],
                  trace.get_values('scale')[idx], trace.get_values('offset')[idx]]
        synthetic_data.append(sim_channel(params))
    synthetic_data = np.array(synthetic_data)
    
    # Train the discriminator
    with tf.GradientTape() as tape:
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(synthetic_data, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    # Update the model's posterior using the discriminator's feedback
    with model:
        pm.set_data({'discriminator_output': fake_output.numpy().mean()})

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(real_data[0, :, 1])
plt.title('Real Data')
plt.subplot(122)
plt.plot(synthetic_data[0, :, 1])
plt.title('Generated Data')
plt.tight_layout()
plt.show()

# Print estimated parameters
print("Estimated parameters:")
for var in ['To', 'Tc', 'Anoise', 'Fnoise', 'scale', 'offset']:
    print(f"{var}: {trace[var].mean():.4f} +/- {trace[var].std():.4f}")
