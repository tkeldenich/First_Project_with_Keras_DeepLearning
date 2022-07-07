# **First Project in Python with Keras – Easy Deep Learning**

*   [English Article](https://inside-machinelearning.com/en/first-projec-keras/)
*   [French Article](https://inside-machinelearning.com/premier-projet-keras/)

If you are new to Deep Learning, this article is for you! Your first Python project with Keras awaits you.

Any beginner in Deep Learning must know Keras.

It is a simple and easy to access library to create your first Neural Networks.

Keras is part of a larger library: TensorFlow.

**By learning how to use Keras you will also learn how to use TensorFlow which will give you the basis to reach a more advanced level.**

Let’s start now this first project with Keras!

## **First Step**

### **Data**

A Deep Learning project is not only done with Keras.

Here we’ll use the Pandas library to load our data. First, let’s import it:


```python
import pandas as pd
```

Now we can load our dataset.

For this project we will use the pima-indian-diabete dataset.

This dataset comes from the National Institute of Diabetes and Digestive and Kidney Diseases.

**The objective of this project is to predict if a patient is diabetic or not, based on diagnostic measures included in the dataset.**

You can download the dataset on [this Github](https://github.com/tkeldenich/First_Project_with_Keras_DeepLearning).

Once you have it, put it in your code environment (Notebook or folder) and load it with the read_csv function. You can then display it with the head function:


```python
df = pd.read_csv('pima-indian-diabetes.csv') 
df.head()
```





  <div id="df-de3399d4-6330-4a52-98d9-22fdbf0f0242">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>DiabetesPresence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-de3399d4-6330-4a52-98d9-22fdbf0f0242')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-de3399d4-6330-4a52-98d9-22fdbf0f0242 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-de3399d4-6330-4a52-98d9-22fdbf0f0242');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We now have our CSV file loaded in a Pandas DataFrame. This DataFrame allows us to easily manipulate our data.

You can see that there are nine columns in total:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Diabetes Presence

Our second step is to extract the X and Y data. Our features and target.

> Features are the data that will allow us to predict our target.

In our case, we want to use the diagnostic data of each patient to predict the presence of diabetes (`DiabetesPresence`).


```python
X = df.loc[:, df.columns != 'DiabetesPresence']
Y = df.loc[:, 'DiabetesPresence']
```

X contains the whole dataset except the column DiabetesPresence. Y contains only the column DiabetesPresence.

In the column `DiabetesPresence` :

- 1 indicates the presence of Diabetes
- 0 indicates the absence of Diabetes.

### **Deep Learning**

Our data is now ready to be used.

We can finally see what Keras is all about!

**Here we import two functions:**

- Sequential to initialize our model
- Dense to add Dense layers to our model


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

We can now use these functions.

First of all we initialize the model. Then we add layers to it:


```python
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

Here the model is quite simple. It has 4 layers:

- Input + Hidden layer: model.add(Dense(12, input_shape=(8,), activation=’relu’))
- Hidden: model.add(Dense(8, activation=’relu’))
- Output: model.add(Dense(1, activation=’sigmoid’))

> Notice that the first layer has in fact two layers.

Indeed the first layer has an input layer indicated by the input_shape of 8. And a hidden layer Dense, in which we indicate the dimension 12 and the activation function relu (more information on these functions at the end of the article).

**For the input and output layers, we must indicate the size of our dataset. In our case, the features have 8 columns. We indicate 8 in the input layer. The target has only one column. Then we indicate 1 in the output layer.**

So our Deep Learning model will take as input our features and will give us as output a prediction of our target.

Then, we can compile our model: we indicate to our model the loss function, the optimizer and the metric to use.

*These are concepts that we will not cover in this tutorial. But keep in mind that these functions are the icing on the cake that allows us to train our Deep Learning model.*


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Finally, we can train our model.

### **Training**

During training, the model will iterate several times on our data. It tries predictions and compares them to the results it should predict. Then it adjusts its weights thanks to the functions defined during the compilation.

These iterations are called epochs. When `epochs=100`, the model will iterate 100 times on the complete dataset.


```python
model.fit(X, Y, epochs=100)
```

    Epoch 1/100
    24/24 [==============================] - 1s 3ms/step - loss: 3.6226 - accuracy: 0.4727
    Epoch 2/100
    24/24 [==============================] - 0s 3ms/step - loss: 2.1144 - accuracy: 0.5768
    Epoch 3/100
    24/24 [==============================] - 0s 3ms/step - loss: 1.4636 - accuracy: 0.5938
    Epoch 4/100
    24/24 [==============================] - 0s 3ms/step - loss: 1.0403 - accuracy: 0.6107
    Epoch 5/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.8284 - accuracy: 0.6263
    Epoch 6/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.7189 - accuracy: 0.6367
    Epoch 7/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.6817 - accuracy: 0.6484
    Epoch 8/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.6731 - accuracy: 0.6445
    Epoch 9/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6652 - accuracy: 0.6523
    Epoch 10/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6511 - accuracy: 0.6510
    Epoch 11/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6450 - accuracy: 0.6523
    Epoch 12/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6429 - accuracy: 0.6484
    Epoch 13/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.6350 - accuracy: 0.6536
    Epoch 14/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.6351 - accuracy: 0.6536
    Epoch 15/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6309 - accuracy: 0.6523
    Epoch 16/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6275 - accuracy: 0.6536
    Epoch 17/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.6239 - accuracy: 0.6523
    Epoch 18/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6219 - accuracy: 0.6523
    Epoch 19/100
    24/24 [==============================] - 0s 7ms/step - loss: 0.6219 - accuracy: 0.6536
    Epoch 20/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6177 - accuracy: 0.6523
    Epoch 21/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.6125 - accuracy: 0.6549
    Epoch 22/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6108 - accuracy: 0.6536
    Epoch 23/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.6100 - accuracy: 0.6536
    Epoch 24/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.6087 - accuracy: 0.6549
    Epoch 25/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.6049 - accuracy: 0.6549
    Epoch 26/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.6043 - accuracy: 0.6549
    Epoch 27/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.6067 - accuracy: 0.6549
    Epoch 28/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.6064 - accuracy: 0.6549
    Epoch 29/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6052 - accuracy: 0.6523
    Epoch 30/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.6026 - accuracy: 0.6549
    Epoch 31/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6028 - accuracy: 0.6549
    Epoch 32/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.6024 - accuracy: 0.6536
    Epoch 33/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.6009 - accuracy: 0.6549
    Epoch 34/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.6010 - accuracy: 0.6536
    Epoch 35/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.6031 - accuracy: 0.6549
    Epoch 36/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5990 - accuracy: 0.6523
    Epoch 37/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5985 - accuracy: 0.6536
    Epoch 38/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5986 - accuracy: 0.6549
    Epoch 39/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5970 - accuracy: 0.6536
    Epoch 40/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5990 - accuracy: 0.6536
    Epoch 41/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5988 - accuracy: 0.6536
    Epoch 42/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5997 - accuracy: 0.6536
    Epoch 43/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5976 - accuracy: 0.6523
    Epoch 44/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5970 - accuracy: 0.6523
    Epoch 45/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6031 - accuracy: 0.6549
    Epoch 46/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5976 - accuracy: 0.6523
    Epoch 47/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5969 - accuracy: 0.6549
    Epoch 48/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5983 - accuracy: 0.6523
    Epoch 49/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.5973 - accuracy: 0.6549
    Epoch 50/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5962 - accuracy: 0.6549
    Epoch 51/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5967 - accuracy: 0.6536
    Epoch 52/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5955 - accuracy: 0.6536
    Epoch 53/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5950 - accuracy: 0.6523
    Epoch 54/100
    24/24 [==============================] - 0s 6ms/step - loss: 0.5967 - accuracy: 0.6562
    Epoch 55/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5971 - accuracy: 0.6536
    Epoch 56/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5955 - accuracy: 0.6549
    Epoch 57/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5953 - accuracy: 0.6562
    Epoch 58/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5969 - accuracy: 0.6562
    Epoch 59/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5952 - accuracy: 0.6536
    Epoch 60/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.6002 - accuracy: 0.6562
    Epoch 61/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5883 - accuracy: 0.6549
    Epoch 62/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5897 - accuracy: 0.6549
    Epoch 63/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5911 - accuracy: 0.6562
    Epoch 64/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5896 - accuracy: 0.6549
    Epoch 65/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5891 - accuracy: 0.6576
    Epoch 66/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5823 - accuracy: 0.6549
    Epoch 67/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5871 - accuracy: 0.6562
    Epoch 68/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5828 - accuracy: 0.6562
    Epoch 69/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5803 - accuracy: 0.6576
    Epoch 70/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5891 - accuracy: 0.6589
    Epoch 71/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5848 - accuracy: 0.6562
    Epoch 72/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5835 - accuracy: 0.6576
    Epoch 73/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5807 - accuracy: 0.6549
    Epoch 74/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5792 - accuracy: 0.6576
    Epoch 75/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5797 - accuracy: 0.6562
    Epoch 76/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5814 - accuracy: 0.6589
    Epoch 77/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5777 - accuracy: 0.6549
    Epoch 78/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5812 - accuracy: 0.6549
    Epoch 79/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5826 - accuracy: 0.6523
    Epoch 80/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5853 - accuracy: 0.6523
    Epoch 81/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5826 - accuracy: 0.6562
    Epoch 82/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5793 - accuracy: 0.6562
    Epoch 83/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5813 - accuracy: 0.6562
    Epoch 84/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5786 - accuracy: 0.6576
    Epoch 85/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5790 - accuracy: 0.6536
    Epoch 86/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5788 - accuracy: 0.6562
    Epoch 87/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5783 - accuracy: 0.6562
    Epoch 88/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5740 - accuracy: 0.6576
    Epoch 89/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5787 - accuracy: 0.6562
    Epoch 90/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5775 - accuracy: 0.6536
    Epoch 91/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5762 - accuracy: 0.6562
    Epoch 92/100
    24/24 [==============================] - 0s 5ms/step - loss: 0.5764 - accuracy: 0.6549
    Epoch 93/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5757 - accuracy: 0.6536
    Epoch 94/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5754 - accuracy: 0.6549
    Epoch 95/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5783 - accuracy: 0.6536
    Epoch 96/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.5763 - accuracy: 0.6523
    Epoch 97/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5743 - accuracy: 0.6562
    Epoch 98/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5737 - accuracy: 0.6562
    Epoch 99/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.5752 - accuracy: 0.6536
    Epoch 100/100
    24/24 [==============================] - 0s 4ms/step - loss: 0.5738 - accuracy: 0.6549





    <keras.callbacks.History at 0x7f32a22fe290>



The model is now trained!

We can then evaluate its performance on our data. To do so, we use the `evaluate` function.


```python
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
```

    24/24 [==============================] - 0s 2ms/step - loss: 0.5728 - accuracy: 0.6562
    Accuracy: 65.62


Our model has an accuracy of 73.83%. This is not bad but…

…it does not represent the real accuracy of our model.
**texte en gras**
Indeed, in Deep Learning, we want to train an Artificial Intelligence model to be efficient in any situation.

If we evaluate our model with the data on which it has been trained, it will necessarily perform better than on data it has never seen.

**And this is precisely what we are interested in. To know if our Deep Learning model is able to use its knowledge on data it has never seen. If it can, we say that the model is able to generalize.**

In the first part of this article, we used a single dataset for training and evaluating our data. But in reality, a real project isn’t done this way.

Actually this part allowed us to validate that we can:

- use our data
- solve our problem with a Deep Learning model

To go further, I propose you the second part of this article.

There, we’ll see how to validate that our model is able to generalize.

## **Going further**

### **Data**

Our dataset contains 789 rows. In other words, we have the medical data of 789 patients.

As we saw in the first part, we need several types of data: one to train our model and the other to evaluate its performance.

These data have a name:

- Training data: `df_train`
- Test data: `df_test`
Problem? We have only one dataset.

Therefore we will create these two datasets ourselves from our CSV.

A large part of this dataset will be used for training and a smaller part for evaluation.

**We’ll take 80% of this data to train our model and 20% to evaluate it.**

Fortunately Pandas allows us to do this easily:


```python
df_train = df.sample(frac=0.8)
df_test = df.drop(df_train.index)
```

First, we take randomly in our dataset 80% of the rows.

Then, we take the remaining rows for evaluation data.

We can display our training data to verify that our data is randomly picked:


```python
df_train.head()
```





  <div id="df-147fdfe8-160c-42d9-860b-38383ef3297d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>DiabetesPresence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>78</td>
      <td>50</td>
      <td>32</td>
      <td>88</td>
      <td>31.0</td>
      <td>0.248</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>467</th>
      <td>0</td>
      <td>97</td>
      <td>64</td>
      <td>36</td>
      <td>100</td>
      <td>36.8</td>
      <td>0.600</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1</td>
      <td>151</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>26.1</td>
      <td>0.179</td>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>387</th>
      <td>8</td>
      <td>105</td>
      <td>100</td>
      <td>36</td>
      <td>0</td>
      <td>43.3</td>
      <td>0.239</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>438</th>
      <td>1</td>
      <td>97</td>
      <td>70</td>
      <td>15</td>
      <td>0</td>
      <td>18.2</td>
      <td>0.147</td>
      <td>21</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-147fdfe8-160c-42d9-860b-38383ef3297d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-147fdfe8-160c-42d9-860b-38383ef3297d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-147fdfe8-160c-42d9-860b-38383ef3297d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We can see that the indexes on the very left column do not follow each other. This indicates that the dataset has been shuffled before picking df_train.

But why randomly select or data?

It allows us to make sure that the separation performed is not biased. Indeed, the dataset as we have it can be sorted beforehand without us being aware of it.

If for example the patients with diabetes were present only at the beginning, and the patients without diabetes only at the end. This would give us training data containing only diabetics. Not ideal if our Deep Learning model has to learn to generalize.

> To be able to generalize, our model must train on disparate data.

Therefore I invite you to check your dataset before performing any operation on it.

**A good practice is to always do a random separation to make sure you don’t end up with biased data.**

From this new data, we can create the X features and Y target. The process is the same as in the first part:


```python
X_train = df_train.loc[:, df.columns != 'DiabetesPresence']
Y_train = df_train.loc[:, 'DiabetesPresence']
X_test = df_test.loc[:, df.columns != 'DiabetesPresence']
Y_test = df_test.loc[:, 'DiabetesPresence']
```

**Model**

Let’s take the structure of our previous Deep Learning model:


```python
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

We can now display its diagram to understand in more detail the Neural Network we’ve just created.

The diagram of a Neural Network allows us to better understand its structure. To do this we use the `plot_model` function:


```python
from tensorflow.keras.utils import plot_model

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, show_layer_activations=True)
```




![png](Readme_files/Readme_37_0.png)



The diagram allows us to visualize the four layers of our Neural Network, their activation function, their input and their output.

For example, our third layer has a `relu` activation function and has 12 neurons as input and 8 as output.

Now let’s compile our model:


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Finally, let’s move on to the most important part: the training.

**Here we will make some changes regarding to the previous code.**

First of all let’s add the `validation_split` this parameter will separate our training data in 2 sub-datasets:

- training data
- validation data

*Note: validation data is not the same as evaluation data. Validation allows us to validate performance as we train. Evaluation is used to assess the model at the end of the training.*

**These concepts will become clearer throughout this article.**

We also indicate that we want to recover the history of the training. This will allow us to analyze our model in more detail. To do this we indicate `history =` at the beginning of the line.

Let’s start the training:


```python
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=50, batch_size=10)
```

    Epoch 1/50
    50/50 [==============================] - 1s 6ms/step - loss: 3.5232 - accuracy: 0.5682 - val_loss: 1.8448 - val_accuracy: 0.5935
    Epoch 2/50
    50/50 [==============================] - 0s 3ms/step - loss: 1.4388 - accuracy: 0.6497 - val_loss: 1.6078 - val_accuracy: 0.6016
    Epoch 3/50
    50/50 [==============================] - 0s 2ms/step - loss: 1.2592 - accuracy: 0.6273 - val_loss: 1.4638 - val_accuracy: 0.5935
    Epoch 4/50
    50/50 [==============================] - 0s 2ms/step - loss: 1.1423 - accuracy: 0.6619 - val_loss: 1.3112 - val_accuracy: 0.5772
    Epoch 5/50
    50/50 [==============================] - 0s 2ms/step - loss: 1.0667 - accuracy: 0.6395 - val_loss: 1.3087 - val_accuracy: 0.5772
    Epoch 6/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.9412 - accuracy: 0.6741 - val_loss: 1.1680 - val_accuracy: 0.5854
    Epoch 7/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.9176 - accuracy: 0.6599 - val_loss: 1.1191 - val_accuracy: 0.6098
    Epoch 8/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.8328 - accuracy: 0.6843 - val_loss: 1.1666 - val_accuracy: 0.5610
    Epoch 9/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.8465 - accuracy: 0.6680 - val_loss: 1.1750 - val_accuracy: 0.5610
    Epoch 10/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.8116 - accuracy: 0.6782 - val_loss: 0.9965 - val_accuracy: 0.6098
    Epoch 11/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.7606 - accuracy: 0.6782 - val_loss: 1.0319 - val_accuracy: 0.5610
    Epoch 12/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.8106 - accuracy: 0.6517 - val_loss: 0.9564 - val_accuracy: 0.5854
    Epoch 13/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.7152 - accuracy: 0.7047 - val_loss: 0.9167 - val_accuracy: 0.6585
    Epoch 14/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.7181 - accuracy: 0.6762 - val_loss: 0.8133 - val_accuracy: 0.6585
    Epoch 15/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.7196 - accuracy: 0.6864 - val_loss: 0.8784 - val_accuracy: 0.6016
    Epoch 16/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.6875 - accuracy: 0.6782 - val_loss: 0.8901 - val_accuracy: 0.5528
    Epoch 17/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6612 - accuracy: 0.6782 - val_loss: 0.8025 - val_accuracy: 0.6098
    Epoch 18/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6655 - accuracy: 0.6741 - val_loss: 0.8083 - val_accuracy: 0.5935
    Epoch 19/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6555 - accuracy: 0.6741 - val_loss: 0.7498 - val_accuracy: 0.6179
    Epoch 20/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.6252 - accuracy: 0.6884 - val_loss: 0.8231 - val_accuracy: 0.5772
    Epoch 21/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6597 - accuracy: 0.6558 - val_loss: 0.7371 - val_accuracy: 0.6260
    Epoch 22/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.6162 - accuracy: 0.6925 - val_loss: 0.7312 - val_accuracy: 0.6423
    Epoch 23/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.6109 - accuracy: 0.6864 - val_loss: 0.7608 - val_accuracy: 0.5935
    Epoch 24/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6070 - accuracy: 0.6945 - val_loss: 0.7747 - val_accuracy: 0.5854
    Epoch 25/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6409 - accuracy: 0.6741 - val_loss: 0.7036 - val_accuracy: 0.6748
    Epoch 26/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6610 - accuracy: 0.7047 - val_loss: 0.6856 - val_accuracy: 0.6911
    Epoch 27/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6161 - accuracy: 0.7006 - val_loss: 0.6881 - val_accuracy: 0.6911
    Epoch 28/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6327 - accuracy: 0.7026 - val_loss: 0.6814 - val_accuracy: 0.6341
    Epoch 29/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5987 - accuracy: 0.7088 - val_loss: 0.6876 - val_accuracy: 0.6341
    Epoch 30/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.6087 - accuracy: 0.6945 - val_loss: 0.7039 - val_accuracy: 0.6341
    Epoch 31/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5924 - accuracy: 0.6864 - val_loss: 0.6879 - val_accuracy: 0.6504
    Epoch 32/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5945 - accuracy: 0.7047 - val_loss: 0.6916 - val_accuracy: 0.6504
    Epoch 33/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5961 - accuracy: 0.7088 - val_loss: 0.6940 - val_accuracy: 0.6504
    Epoch 34/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.6124 - accuracy: 0.7026 - val_loss: 0.7350 - val_accuracy: 0.6829
    Epoch 35/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5892 - accuracy: 0.7169 - val_loss: 0.6607 - val_accuracy: 0.6585
    Epoch 36/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5766 - accuracy: 0.7373 - val_loss: 0.6484 - val_accuracy: 0.6667
    Epoch 37/50
    50/50 [==============================] - 0s 4ms/step - loss: 0.6112 - accuracy: 0.6884 - val_loss: 0.6473 - val_accuracy: 0.6992
    Epoch 38/50
    50/50 [==============================] - 0s 5ms/step - loss: 0.5816 - accuracy: 0.6945 - val_loss: 0.6991 - val_accuracy: 0.6992
    Epoch 39/50
    50/50 [==============================] - 0s 4ms/step - loss: 0.5864 - accuracy: 0.7026 - val_loss: 0.6525 - val_accuracy: 0.6829
    Epoch 40/50
    50/50 [==============================] - 0s 4ms/step - loss: 0.5811 - accuracy: 0.7047 - val_loss: 0.6647 - val_accuracy: 0.7398
    Epoch 41/50
    50/50 [==============================] - 0s 5ms/step - loss: 0.5888 - accuracy: 0.7149 - val_loss: 0.6759 - val_accuracy: 0.6911
    Epoch 42/50
    50/50 [==============================] - 0s 4ms/step - loss: 0.5805 - accuracy: 0.7088 - val_loss: 0.6605 - val_accuracy: 0.7073
    Epoch 43/50
    50/50 [==============================] - 0s 5ms/step - loss: 0.5799 - accuracy: 0.7026 - val_loss: 0.6449 - val_accuracy: 0.6748
    Epoch 44/50
    50/50 [==============================] - 0s 5ms/step - loss: 0.6443 - accuracy: 0.6904 - val_loss: 0.6478 - val_accuracy: 0.7154
    Epoch 45/50
    50/50 [==============================] - 0s 4ms/step - loss: 0.5835 - accuracy: 0.7108 - val_loss: 0.6947 - val_accuracy: 0.6911
    Epoch 46/50
    50/50 [==============================] - 0s 4ms/step - loss: 0.5694 - accuracy: 0.7251 - val_loss: 0.6589 - val_accuracy: 0.6585
    Epoch 47/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5614 - accuracy: 0.7189 - val_loss: 0.6514 - val_accuracy: 0.6667
    Epoch 48/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5764 - accuracy: 0.7189 - val_loss: 0.6807 - val_accuracy: 0.6260
    Epoch 49/50
    50/50 [==============================] - 0s 2ms/step - loss: 0.5676 - accuracy: 0.7271 - val_loss: 0.6415 - val_accuracy: 0.6667
    Epoch 50/50
    50/50 [==============================] - 0s 3ms/step - loss: 0.5635 - accuracy: 0.7251 - val_loss: 0.6163 - val_accuracy: 0.6911


### **Analysis**

Once the training is finished, we can visually analyze what happened during this process.

For this we use the Matplotlib library:


```python
from matplotlib import pyplot as plt
```

Our `history` variable has recorded all the information about the model training.

At each epoch the model :

- makes predictions on training AND validation data
- compares its predictions on training AND validation data with the expected result
- changes its weights according to its performance on training data ONLY

The validation data are solely present to validate the results of the model on external data. Those on which it does not train.

In fact, this allows to counter what is called overfitting.

> Overfitting occurs when the model specializes too much on the training data.

**It then performs well on training data but fails to predict good results on the other data.**

The validation data are used to check that this does not happen.

Let’s display the results:


```python
plt.plot(history.history['accuracy'], color='#066b8b')
plt.plot(history.history['val_accuracy'], color='#b39200')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![png](Readme_files/Readme_46_0.png)


In blue the results on training data. In orange the results on validation data.

If the orange curve follows the blue curve the training is well done. When the model increases its performance on the training data, it also increases its performance on the validation data. This indicates that the model generalizes well.

Visually we can see the overfitting when the validation curve deviates from the training curve. Here we can see that it starts to deviate from epoch 45 to come back on the right path at epoch 50.

**Keep in mind that there will always be a gap between the results of the training curve and those of the validation curve. This is because the model is trained explicitly on the training data. It therefore performs better in this context.**

Our goal is to keep this gap as small as possible and keep the curves in line.

At epoch 50, the accuracy gap between the training and validation data is less than 0.5. This is considered acceptable.

If this wasn’t the case, we would have had to rerun the training with other parameters.

### **Predictions**

**Now how to make predictions on new data?**

We can reuse our model to make it predict the presence of diabetes on new data. For that nothing is easier, we use the `predict``
 function:


```python
predictions = model.predict(X_test)
```

We get an array of predictions.

Let’s display the first element:


```python
predictions[0]
```




    array([0.7407335], dtype=float32)



Here we get neither 1 nor 0 but a decimal 0.74. In fact it is a probability.

**We can establish this rule: if the probability is above 50%, there is diabetes otherwise there is no diabetes.**

Let’s modify the result of these predictions to have only 0 and 1:


```python
predictions = (model.predict(X_test) > 0.5).astype(int)
```

Now, instead of having probabilities we have 0’s and 1’s representing the presence of diabetes according to the rule established above.

We can display the first predictions and compare them to the expected results:


```python
for i in range(5):
	print('%s => Predicted : %d,  Expected : %d' % (X_test.iloc[i].tolist(), predictions[i], Y_test.iloc[i]))
```

    [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0] => Predicted : 1,  Expected : 1
    [10.0, 115.0, 0.0, 0.0, 0.0, 35.3, 0.134, 29.0] => Predicted : 1,  Expected : 0
    [2.0, 197.0, 70.0, 45.0, 543.0, 30.5, 0.158, 53.0] => Predicted : 1,  Expected : 1
    [1.0, 189.0, 60.0, 23.0, 846.0, 30.1, 0.398, 59.0] => Predicted : 1,  Expected : 1
    [7.0, 107.0, 74.0, 0.0, 0.0, 29.6, 0.254, 31.0] => Predicted : 0,  Expected : 1


**The model makes good predictions on most of the cases.**

We could continue to evaluate line by line each prediction of our model.

But we will prefer the function included in Keras evaluate to determine its performance:


```python
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))
```

    5/5 [==============================] - 0s 3ms/step - loss: 0.5607 - accuracy: 0.7273
    Accuracy: 72.73


The result is quite good. We are very close to the value obtained in the first part of 73.83. This indicates that the model succeeds in generalizing.



## **To conclude**

You have developed your first Neural Network with Keras and with an impressive prediction of 72%!

> Will you be able to exceed this result?

To do so, several techniques are offered to you:

- Add layers
- Modify [the activation functions](https://inside-machinelearning.com/en/activation-function-how-does-it-work-a-simple-explanation/)
- Use other functions at compile time
- Modify the training parameters of the model

And keep in mind to ensure that your model is not overfitting!

If you want to continue learning, our article on activation functions is waiting for you here: [Activation function, how does it work ? – A simple explanation.](https://inside-machinelearning.com/en/activation-function-how-does-it-work-a-simple-explanation/)


```python

```
