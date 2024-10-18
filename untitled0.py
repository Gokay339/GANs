# Eğitim setiyle aynı istatistiklere sahip yeni veriler oluşturmayı öğrenen yapılardır


# İlk adımda üretici, rastgele bir girdi alır ve bir çıktı (örneğin, bir görüntü) üretir.
# Bu çıktı ayırt ediciye verilir, ayırt edici ise bu çıktının gerçek bir veri mi yoksa üretici 
# tarafından üretilmiş sahte bir veri mi olduğunu anlamaya çalışır.
# Ayırt edici, gerçek verilerle sahte verileri doğru bir şekilde ayırmayı öğrenmeye çalışırken, 
# üretici de ürettiği verilerin gerçekmiş gibi algılanmasını sağlamaya çalışır. 
# Bu şekilde iki ağ birbirlerine karşı bir yarış içinde eğitilir.


# 2 Networkvar

# 1 Üretici Ağ (Generator) : Kalpazan deniyor (sahte şeyler üreten)    
# 2 Discriminator (Ayırt Edici Ağ): Ayırt ettiği için dedektif deniyor genelde

# Asıl amaç gerçeğinden ayırt edilemez datalar üretmek. Generetor yapıcak
# Generetor data üreticek ve Discriminator kontrol edicek

# Gans eğitim setiyle yeni veriler üreticek


from keras.layers import Dense , Dropout,Input,ReLU
from keras.models import Model , Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test) = mnist.load_data()
# mnist 0dan 9 kadar 0 dahil  yani 10 tane class var

x_train =(x_train.astype(np.float32)-127.5)/127.5   # -1 ve 1 arasında ölçeklendirmek anlamına gelir
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
print(x_train.shape)


#%%

plt.imshow(x_test[1]) 


#%% generator oluşturma


def create_generator():
    generator = Sequential()

    generator.add(Dense(units=512,input_dim=100))
    #   512 Nöron koyuyoruz , giriş verisinin boyutu 100
    generator.add(ReLU())
    # Relu : Bu fonksiyon, sinir ağlarındaki nöronlar için girişleri doğrusal
    # olmayan bir hale getirmek için kullanılır.
    
    generator.add(Dense(units=512))
    generator.add(ReLU())
        
    generator.add(Dense(units=1024))
    generator.add(ReLU())
        
    generator.add(Dense(units=784,activation="tanh"))

    generator.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=0.0001,beta_1=0.5))
    return generator

g = create_generator()
g.summary() # Yarattığımız katmanları görme


#%% descriminator dedektif 

def create_descriminator():
    descriminator = Sequential()
    
    descriminator.add(Dense(units=1024,input_dim=784))
    descriminator.add(ReLU())
    descriminator.add(Dropout(0.4))
    
    descriminator.add(Dense(units=256))
    descriminator.add(ReLU())
    
    descriminator.add(Dense(units=1,activation="sigmoid"))
    descriminator.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=0.0001,beta_1=0.5))
    return descriminator
d = create_descriminator()
d.summary()
    



#%% gans

def create_gan(descriminator,generator):
    descriminator.trainabe = False # EĞİTİLMEZ YAPIYORUZ
    
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = descriminator(x)
    
    gan = Model(inputs=gan_input,outputs= gan_output )
    gan.compile(loss="binary_crossentropy",
                      optimizer="adam")
    return gan
gan = create_gan(d,g)
gan.summary()

#%% train

epochs = 50
batch_size = 256

for i in range(epochs):
    for _ in range(batch_size):
        
        noise = np.random.normal(0,1,[batch_size,100])
        generated_images = g.predict(noise)
        
        image_batch = x_train[np.random.randint(low=0 , high=x_train.shape[0],size=batch_size)]
        
        x = np.concatenate([image_batch,generated_images])
        
        y_dis = np.zeros(batch_size*2)
        y_dis[:batch_size]=1
        
        d.trainable = True
        d.train_on_batch(x,y_dis)
        
        
        noise = np.random.normal(0,1,[batch_size,100])
        y_gen = np.ones(batch_size)
        d.trainable = False
        
        gan.train_on_batch(noise,y_gen)
        
    print("Epochs : ",i)
        




































