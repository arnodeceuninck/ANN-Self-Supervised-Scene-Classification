# Submission feedback
Feedback provided by Hamed Behzadi Khormouji.


1. In whole of your report, you frequently refer to “multi-label classification tasks”. According to what we explained in the project description, all the tasks are “multi-class classification”. I’ve read your report multiple times to figure out how did you define the labels and do the multi-label classification. However, there wasn’t any explanation in this regard. For more info, I would like to refer you to the second practical session where we discuss this.


2. In table 2 you have reported some values as Accuracy. However, you didn’t mention these obtained values are train or test classification accuracy. You needed to report both.


3. In the comparison section, you have repeated those accuracies demonstrated in Table 2 without proper justification on the results and comparison the models with respect to the accuracies and confusion matrices.


4. The description provided about the models' classification performance is not clear. For example, you have mentioned “the features from efficientnet after the pretext task were not useful enough to freeze them and only rely on the classification layer.” So, here I was unable to understand what you mean by not useful to freeze them. If those mentioned features are not useful, you didn’t mention what those features could be. Moreover, in the pretext tasks, the models are not only relying on the features from fully-connected layer, but also, the extracted features from convolutional layers. 


5. In section Overfitting, you have explained the ways you tried to avoid overfitting over epochs. However, you didn’t provide any train and validation loss plot or train and validation accuracy plot as evidence to not being trapped in overfitting or underfitting.


6. About the explanation parts. Some points are as follows. (1) The visualizations have not been generated properly. It is not clear what do colors mean in your heatmaps as you used matplotlib imshow function. According to the practical session, we use color map “Jet” to assign color to a map such that the color appeared on the visualization has a defined range of colors. Moreover, there was not any procedure to assign color map to the heatmap as we did it in practical session. (2) Two of the figures have the same captions, so it is not clear which one is related to rotation task and which one for perturbation task. (4) you have mentioned that
“The model after the perburbation pretext task looks more at objects more spread over the image (often the borders of the image). This is probably because the perbutations could block the actual part of interest when only looking at one specific location.”
Actually, this justification may not be correct. First, from your visualizations I think by “more spread over the image” you mean the scene. Second, this observation shows that the models trained on top of the perturbation pretext task were able to reuse the scene features learned during the perturbation pretext. That's why you see that visualizations highlights the scene features (3) Finally, you were expected to discuss the visualizations from different models w.r.t each other for example whether the models trained in the self-supervised scheme were able to reuse the scene-related features learned during the pretext task.


7. About the Interpretation part, I would like to give the following feedback. You mentioned that “the interpretation algorithm does not work on the models you have trained, and the gradients did not get calculated.” Well, I would like to refer to the last practical session where we discussed the procedure of the hyper-parameter tuning for the interpretation method. Since your trained model has different representations than the model pre-trained on ImageNet, you need modify hyper-parameters of the algorithm to be specific to your trained model. Also, one of the hyper-parameters is learning rate that by adjusting it you can monitor the training procedure of the interpretation algorithm.