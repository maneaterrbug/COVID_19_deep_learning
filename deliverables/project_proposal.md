# Applying CNN to lung CT scans to predict infection with COVID-19
Matt Ryan

## Proposal
In my project,I intend to use convolutional neural networking on a series of CT lung scan datasets to attempt to predict whether or not a patient is infected with COVID-19. The dataset that inspired this project can be found [here](https://github.com/UCSD-AI4H/COVID-CT), and contains 349 total CT scans, of which 216 contain clinical findings of COVID-19. For more information about how this set was gathered, see the aforementioned repository's read-me. In addition, should the model training require additional data, I may include [this set](https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-021-05592-x), which contains over 1000 scans of COVID-19 positive patients. It bares mention that should this be the case, I will also need to bring in additional CT scans of lungs in healthy, COVID-free individual to combat class imbalance.

## MVP
A minimum viable product for this project will be a CNN trained model that can make class predictions on CT scans of lungs as well as model evaluation and associated evaluation metrics.
