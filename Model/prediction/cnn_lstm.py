import torch
import torch.nn as nn
import numpy as np
from pandas import DataFrame, date_range


class CNN_LSTM(nn.Module):
    """
    A combined CNN-LSTM model for time-series prediction and their classification.

    This model uses a CNN for image feature extraction (embedding) and an LSTM for temporal
    sequence processing. It predicts future agronomic characteristics and classifies the time series.

    Parameters
    ----------
    CNN : torch.nn.Module
        Convolutional Neural Network for image embedding.
    num_layers : int
        Number of LSTM layers.
    steps_pred : int, optional
        Number of time steps to predict ahead. Default is 15.
    step_input : int, optional
        Number of input time steps. Default is 15.
    num_para : int, optional
        Number of agronomic characteristics to predict. Default is 4.
    bidir : bool, optional
        If True, uses a bidirectional LSTM. Default is False.

    Attributes
    ----------
    steps_pred : int
        Number of time steps to predict.
    step_input : int
        Number of input time steps.
    num_para : int
        Number of agronomic characteristics.
    bidir : bool
        Whether the LSTM is bidirectional.
    cnn : torch.nn.Module
        CNN for image embedding.
    para_cnn : torch.nn.Linear
        Linear layer to simplify CNN embeddings.
    lstm : torch.nn.LSTM
        LSTM for temporal sequence processing taking the simplified CNN embeddings,
        to give an embedding of images which takes into account temporal information.
    predict_lstm : torch.nn.Linear
        Linear layer for predicting future agronomic characteristics using all lstm embedding.
    classif_lstm : torch.nn.Sequential
        Sequential network for classifying time series using the last lstm embedding.
    """

    def __init__(self, CNN, num_layers, steps_pred=15, step_input=15, num_para=4, bidir=False):
        # Initialize the parent class.
        super(CNN_LSTM, self).__init__()

        ## Store model characteristics.
        self.steps_pred = steps_pred
        self.step_input = step_input
        self.num_para = num_para
        self.bidir = bidir

        ## Intialize the different block of the algorithm.
        # Initialize CNN for image embedding.
        self.cnn = CNN
        # Linear layer to simplify CNN embeddings.
        self.para_cnn = nn.Linear(5*135*240, num_para)
        # LSTM for temporal sequence processing.
        self.lstm = nn.LSTM(
            input_size=num_para,
            hidden_size=num_para,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir
        )
        # Linear layer for predicting future agronomic characteristics.
        self.predict_lstm = nn.Linear(
            in_features=num_para*step_input,
            out_features=num_para*steps_pred
        )
        # Sequential network for classifying time series.
        self.classif_lstm = nn.Sequential(
            nn.Linear(in_features=num_para, out_features=3)
        )
        self.name = "CNN_LSTM"

    def forward(self, x):
        """
        Forward pass of the CNN-LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, length, channels, image_x, image_y).

        Returns
        -------
        tuple
            A tuple containing:
            - out_predict: Predicted agronomic characteristics of shape (batch_size, steps_pred, num_para)
            - out_class: Class predictions of shape (batch_size, 3)
        """
        ## CNN Feature Extraction
        # Set CNN to evaluation mode (no gradient updates).
        self.cnn.eval()
        # Initialize list to store all simplified CNN embedings.
        seq_out = []
        bath_size = x.shape[0]
        # Process each batch in the input.
        for k in x:
            with torch.no_grad():
                ## Creating a CNN embeding of images
                # Remove zero-sum class mask (mask with no entities) and track their positions.
                new_k = []
                date_r_in = list(date_range(start="07-31-2000", periods=k.shape[0]))
                date_r_out = list(date_range(start="07-31-2000", periods=k.shape[0]))
                corr = 0
                for l in range(k.shape[0]):
                    if k[l].sum() != 0:
                        new_k.append(k[l])
                    else:
                        date_r_out.pop(l-corr)
                        corr += 1
                # Convert the filtered images to a tensor.
                new_k = torch.tensor(np.array(new_k)).float()
                # Extract features using CNN.
                out = self.cnn(new_k)
                # Handle output format for specific CNN architectures.
                if self.cnn.name == "deeplab3" or self.cnn.name == "Mobilenetv3":
                    out = out["out"]
            
            ## Simplify the CNN embeding
            # Reshape CNN output for the linear layer.
            out = out.reshape((out.shape[0], out.shape[1]*out.shape[2]*out.shape[3]))
            # Simplify the CNN embeding.
            out = self.para_cnn(out)

            with torch.no_grad():
                ## Handle Missing date (images)
                # Create a DataFrame with the CNN embedings.
                out_time = DataFrame(out, index=date_r_out)
                # Reindex to the original time range and interpolate missing CNN embedings.
                out_time = out_time.reindex(date_r_in)
                out_time = out_time.infer_objects(copy=False).interpolate(
                    method="time",
                    limit_direction='both'
                )
                # Convert back to numpy array.
                new_out = np.array(out_time)
                seq_out.append(new_out)

        ## LSTM Processing
        # Convert sequence outputs to a tensor.
        seq_out = torch.tensor(np.array(seq_out))
        # Pass through LSTM.
        output_all, (h_n, _) = self.lstm(seq_out)

        ## Prediction of Agronomic Parameters
        if self.bidir:
            # For bidirectional LSTM, combine forward and backward outputs.
            output_all = output_all[:, :, :self.num_para] + output_all[:, :, self.num_para:]
        # Reshape LSTM output for prediction.
        print(output_all.shape)
        predict_in = output_all.reshape(bath_size, self.step_input*self.num_para)
        # Predict future agronomic characteristics.
        out_predict = self.predict_lstm(predict_in)
        out_predict = out_predict.reshape((bath_size, self.steps_pred, self.num_para))

        ## Classification of Time Series
        if self.bidir:
            # For bidirectional LSTM, combine the final forward and backward hidden states.
            h_n_class = h_n[-2:]  # Get the last layer's forward and backward states
            h_n_class = h_n_class[0] + h_n_class[1]
        else:
            # Use the final hidden state for classification.
            h_n_class = h_n[-1]
        # Classify the time series.
        out_class = self.classif_lstm(h_n_class)

        return out_predict, out_class

class all_train_CNN_LSTM(nn.Module):
    def __init__(self, CNN, num_layers, steps_pred = 15, step_input = 15, num_para = 4, bidir = False):
        super(all_train_CNN_LSTM, self).__init__()
        self.steps_pred = steps_pred
        self.step_input = step_input
        self.num_para = num_para
        self.num_layers = num_layers
        self.bidir = bidir
        self.name = "all_train_CNN_LSTM"

        self.cnn = CNN # Segmentation des images => Ici plutot embeding des images dans un espace representatif
        self.para_cnn = nn.Linear(5*135*240, num_para) # Extraction des paramètres agronomiques des représentation (image)
        self.lstm = nn.LSTM(input_size=num_para, hidden_size=num_para, num_layers=num_layers, batch_first=True, bidirectional = bidir) # Représentation des paramètres en série temporelle
        
        self.predict_lstm = nn.Linear(in_features=num_para*step_input, out_features=num_para*steps_pred) # Prediction des paramètre agro
        self.classif_lstm = nn.Sequential(nn.Linear(in_features=num_para, out_features=3)) # Classification de la série temporelle)
        
    def forward(self, x):
        #Input shape (batch_size, length, channels, image_x, image_y)
        seq_out = [] 
        bath_size = x.shape[0]
        for k in x: # boucle sur chaque batch
            new_k = []
            date_r_in = list(date_range(start= "07-31-2000", periods=k.shape[0]))
            date_r_out = list(date_range(start= "07-31-2000", periods=k.shape[0]))
            corr = 0
            for l in range(k.shape[0]):
                if k[l].sum() != 0:
                    new_k.append(k[l])
                else:
                    date_r_out.pop(l-corr)
                    corr +=1
            new_k = torch.tensor(np.array(new_k)).float()
            out = self.cnn(new_k) # input shape (length, channels, image_x, image_y) # output shape (length, parameters)
            if self.cnn.name == "deeplab3" or self.cnn.name == "Mobilenetv3":
                out = out["out"]
            
            out = out.reshape((out.shape[0],out.shape[1]*out.shape[2]*out.shape[3]))
            out = self.para_cnn(out)

            #    out = out.reshape((out.shape[0],out.shape[1]*out.shape[2]*out.shape[3])) # Get rid of the linear (don't forget change the input shape of lstm)
            
            with torch.no_grad():
                # Filling the gapes of image 
                out_time = DataFrame(out, index=date_r_out)
                out_time = out_time.reindex(date_r_in)
                out_time = out_time.infer_objects(copy=False).interpolate(method="time", limit_direction='both')
                new_out = np.array(out_time)

                seq_out.append(new_out) # Après la boucle : (batch_size, length, parameters)
        
        # lstm takes input of shape (batch, seq_len, parameters)
        seq_out = torch.tensor(np.array(seq_out))
        output_all, (h_n,_) = self.lstm(seq_out)
        # lstm returns an output of shape (batch, seq_len, parameters)

        # Decoder pour prédire les prochains paramètres 
        if self.bidir :
            output_all_temp = output_all[:,:,:self.num_para] + output_all[:,:,:self.num_para]
            predict_in = output_all_temp.reshape(bath_size,self.step_input*self.num_para)
        else:
            predict_in = output_all.reshape(bath_size,self.step_input*self.num_para) # reshape that respect the sequence of values (first one is the batch 1 len 1 para 1) 
        out_predict = self.predict_lstm(predict_in)
        out_predict = out_predict.reshape((bath_size,self.steps_pred, self.num_para))


        # Decoder pour sortir la classe
        if self.bidir :
            h_n_class = h_n[-1] + h_n[-2] # Prendre la dernier couche forward et backward
        else:
            h_n_class = h_n[-1] # Prendre la dernier couche

        # input : (batch, seq_len, hidden_state)

        # h_n_class = h_n[self.num_layers:,:,:] + h_n[:self.num_layers,:,:]
        # h_n_class = torch.moveaxis(h_n_class, 0, 1)
        # h_n_class = h_n_class.reshape(3, self.num_layers*self.num_para)

        out_class = self.classif_lstm(h_n_class)
        # output : (batch, class))
        return out_predict, out_class

class no_CNN_LSTM(nn.Module):
    def __init__(self, num_layers, steps_pred = 15, step_input = 15, num_para = 4, bidir = False):
        super(no_CNN_LSTM, self).__init__()
        self.steps_pred = steps_pred
        self.step_input = step_input
        self.num_para = num_para
        self.num_layers = num_layers
        self.bidir = bidir
        self.name = "no_CNN_LSTM"

        # self.cnn = CNN # Segmentation des images => Ici plutot embeding des images dans un espace representatif
        self.para_cnn = nn.Linear(3*1920*1080, num_para) # Extraction des paramètres agronomiques des représentation (image)
        self.lstm = nn.LSTM(input_size=num_para, hidden_size=num_para, num_layers=num_layers, batch_first=True, bidirectional = bidir) # Représentation des paramètres en série temporelle
        
        self.predict_lstm = nn.Linear(in_features=num_para*step_input, out_features=num_para*steps_pred) # Prediction des paramètre agro
        self.classif_lstm = nn.Sequential(nn.Linear(in_features=num_para, out_features=3)) # Classification de la série temporelle)
        
    def forward(self, x):
        #Input shape (batch_size, length, channels, image_x, image_y)
        seq_out = [] 
        bath_size = x.shape[0]
        for k in x: # boucle sur chaque batch
            new_k = []
            date_r_in = list(date_range(start= "07-31-2000", periods=k.shape[0]))
            date_r_out = list(date_range(start= "07-31-2000", periods=k.shape[0]))
            corr = 0
            for l in range(k.shape[0]):
                if k[l].sum() != 0:
                    new_k.append(k[l])
                else:
                    date_r_out.pop(l-corr)
                    corr +=1
            new_k = torch.tensor(np.array(new_k)).float()
            # out = self.cnn(new_k) # input shape (length, channels, image_x, image_y) # output shape (length, parameters)
            # if self.cnn.name == "deeplab3" or self.cnn.name == "Mobilenetv3":
            #     out = out["out"]
            
            # out = out.reshape((out.shape[0],out.shape[1]*out.shape[2]*out.shape[3]))
            out = new_k.reshape(new_k.shape[0], new_k.shape[1]* new_k.shape[2] * new_k.shape[3] )
            out = self.para_cnn(out)

            #    out = out.reshape((out.shape[0],out.shape[1]*out.shape[2]*out.shape[3])) # Get rid of the linear (don't forget change the input shape of lstm)
            
            with torch.no_grad():
                # Filling the gapes of image 
                out_time = DataFrame(out, index=date_r_out)
                out_time = out_time.reindex(date_r_in)
                out_time = out_time.infer_objects(copy=False).interpolate(method="time", limit_direction='both')
                new_out = np.array(out_time)

                seq_out.append(new_out) # Après la boucle : (batch_size, length, parameters)
        
        # lstm takes input of shape (batch, seq_len, parameters)
        seq_out = torch.tensor(np.array(seq_out))
        output_all, (h_n,_) = self.lstm(seq_out)
        # lstm returns an output of shape (batch, seq_len, parameters)

        # Decoder pour prédire les prochains paramètres 
        if self.bidir :
            output_all_temp = output_all[:,:,:self.num_para] + output_all[:,:,:self.num_para]
            predict_in = output_all_temp.reshape(bath_size,self.step_input*self.num_para)
        else:
            predict_in = output_all.reshape(bath_size,self.step_input*self.num_para) # reshape that respect the sequence of values (first one is the batch 1 len 1 para 1) 
        out_predict = self.predict_lstm(predict_in)
        out_predict = out_predict.reshape((bath_size,self.steps_pred, self.num_para))


        # Decoder pour sortir la classe
        if self.bidir :
            h_n_class = h_n[-1] + h_n[-2] # Prendre la dernier couche forward et backward
        else:
            h_n_class = h_n[-1] # Prendre la dernier couche

        # input : (batch, seq_len, hidden_state)

        # h_n_class = h_n[self.num_layers:,:,:] + h_n[:self.num_layers,:,:]
        # h_n_class = torch.moveaxis(h_n_class, 0, 1)
        # h_n_class = h_n_class.reshape(3, self.num_layers*self.num_para)

        out_class = self.classif_lstm(h_n_class)
        # output : (batch, class))
        return out_predict, out_class

