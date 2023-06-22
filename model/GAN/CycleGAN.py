from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2DTranspose, LeakyReLU,Conv2D,LeakyReLU,ReLU
from tensorflow import pad,GradientTape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import ones_like,zeros_like
from tensorflow.keras.initializers import he_normal

# Loss function for evaluating adversarial loss

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = MeanSquaredError(ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = MeanSquaredError(ones_like(real), real)
    fake_loss = MeanSquaredError(zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


class ReflectionPadding2D(Layer):
    """Implements Reflection Padding as a layer.
    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.
    Returns:
        A padded tensor with the same type as the input tensor.
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return pad(input_tensor, padding_tensor, mode="REFLECT")

class ResidualBlock(Layer):
    def init(
    self,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
    **kwargs
    ):
        super(ResidualBlock, self).init(**kwargs)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.gamma_initializer = gamma_initializer
        self.use_bias = use_bias

    def build(self, input_shape):
        dim = input_shape[-1]
        self.padding1 = ReflectionPadding2D()
        self.conv1 = Conv2D(
            dim,
            self.kernel_size,
            strides=self.strides,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            use_bias=self.use_bias,
        )
        self.instance_norm1 = InstanceNormalization(
            gamma_initializer=self.gamma_initializer
        )
        self.padding2 = ReflectionPadding2D()
        self.conv2 = Conv2D(
            dim,
            self.kernel_size,
            strides=self.strides,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            use_bias=self.use_bias,
        )
        self.instance_norm2 = tfa.layers.InstanceNormalization(
            gamma_initializer=self.gamma_initializer
        )
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs):
        x = self.padding1(inputs)
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = self.activation(x)

        x = self.padding2(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)

        return tf.keras.layers.add([inputs, x])

class Downsample(Layer):
    def init(self,filters,activation,kernel_initializer,
             kernel_size=(3, 3),strides=(2, 2),padding="same",
             gamma_initializer,
             use_bias=False,**kwargs):
        super(Downsample, self).init(**kwargs)
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.gamma_initializer = gamma_initializer
        self.use_bias = use_bias

        self.conv = Conv2D(
                self.filters,
                self.kernel_size,
                strides=self.strides,
                kernel_initializer=self.kernel_initializer,
                padding=self.padding,
                use_bias=self.use_bias,
            )
        self.instance_norm = tfa.layers.InstanceNormalization(gamma_initializer=self.gamma_initializer)
            
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Upsample(Layer):
    def init(self,filters,activation,kernel_size=(3, 3),strides=(2, 2),padding="same",kernel_initializer,gamma_initializer,use_bias=False,**kwargs):
        super(Upsample, self).init(**kwargs)
        self.filters = filters
        self.activation = activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer
        self.use_bias = use_bias

    def build(self, input_shape):
        self.conv_transpose = Conv2DTranspose(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias,
        )
        self.instance_norm = tfa.layers.InstanceNormalization(
            gamma_initializer=self.gamma_initializer
        )
    super(Upsample, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class ResNetGenerator(Model):
    def init(self,filters=64,num_downsampling_blocks=2,num_residual_blocks=9,num_upsample_blocks=2,gamma_initializer=gamma_init,name=None):
        super(ResNetGenerator, self).init(name=name)
        self.filters = filters
        self.num_downsampling_blocks = num_downsampling_blocks
        self.num_residual_blocks = num_residual_blocks
        self.num_upsample_blocks = num_upsample_blocks
        self.gamma_initializer = gamma_initializer
        self.reflection_padding = ReflectionPadding2D(padding=(3, 3))
        self.conv1 = Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)
        self.instance_norm1 = tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_initializer
        )
        self.activation1 = ReLU()

        # Downsampling
        self.downsampling_blocks = []
        for _ in range(num_downsampling_blocks):
            filters *= 2
            self.downsampling_blocks.append(Downsample(
                filters=filters,
                activation=ReLU()
            ))

        # Residual blocks
        self.residual_blocks = [ResidualBlock(activation=ReLU()) for _ in range(num_residual_blocks)]


        # Upsampling
        self.upsampling_blocks = []
        for _ in range(num_upsample_blocks):
            filters //= 2
            self.upsampling_blocks.append(Upsample(
                filters=filters,
                activation=ReLU()
            ))

        self.reflection_padding2 = ReflectionPadding2D(padding=(3, 3))
        self.conv2 = Conv2D(3, (7, 7), padding="valid",activation="tanh")

    def call(self, inputs):
        x = self.reflection_padding(inputs)
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = self.activation1(x)

        for block in self.downsampling_blocks:
            x = block(x)

        for block in self.residual_blocks:
            x = block(x)

        for block in self.upsampling_blocks:
            x = block(x)

        x = self.reflection_padding2(x)
        x = self.conv2(x)
        return x
    
class Discriminator(Model):
    def __init__(self, filters=64, kernel_initializer=he_normal(), num_downsampling=3, name=None):
        super(Discriminator, self).__init__(name=name)
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.num_downsampling = num_downsampling

        self.conv1 = Conv2D(
            filters,
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )
        self.leaky_relu = LeakyReLU(0.2)

        self.downsamples = []
        for num_downsample_block in range(num_downsampling):
            num_filters *= 2
            strides= (2, 2) if num_downsample_block < 2 else (1, 1)
            self.downsamples.append(Downsample(
                    filters=num_filters,
                    activation=LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=strides,
                ))

        self.final_conv = Conv2D(
            1,
            (4, 4),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky_relu(x)

        for downsample_layer in self.downsamples:
            x = downsample_layer(x)

        x = self.final_conv(x)

        return x


class CycleGan(Model):
    def __init__(
        self,
        generator_G=ResNetGenerator(name="generator_G"),
        generator_F=ResNetGenerator(name="discriminator_F"),
        discriminator_X=Discriminator(name="discriminator_X"),
        discriminator_Y=Discriminator(name="discriminator_Y"),
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity


    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = MeanSquaredError()
        self.identity_loss_fn = MeanSquaredError()



    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
