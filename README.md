# project-1 (In Matlab)
forecast generation of solar induction on clear and cloudy day
this project uses solar irradiation and generation data of one year taken at every 15 minute interval.


aim of this project was to predict the solar generation data based on the genertaion and irradtion data available on clear and cloudy day. 

we can analyse the graphs of the data after preprocessing and diffrentiate the clear days from cloudy days based on the shape and the irregularities of the graph.

thus we can extract datapoints from the data set and maintain two sets of 1000 points of each clear and cloudy day.

creating further datasets from these by picking every 2nd, 3rd and 4th point. this gives us the data at every 30 mins,45mins and 60 mins respectively.
 
then using a lstm model we can feed the data and train the model and predict the  forecast of solar generation at every 15min,30min,45min,60min on a clear and cloudy day.

i am enclosing the datasets along with my code in matlab

