//! A simple implementation of a neural network  
//! 
//! Based on [ttps://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html]
//! 
//! [ttps://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html]: https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html


// pub struct SomeData {
//     pub toast: String,
//     fruit: String,
// }

/// Eat at a restaurant  
/// [Rust website]
/// 
/// [Rust website]: http://www.rust-lang.org
/// 
/// The [`String`] passed in lorum ipsum...
///
/// [`String`]: ./src/main.rs
///
/// Arguments
/// ---------
///
/// * `name` - A string slice that holds the name of the person
///
/// Examples
/// --------
///
/// ```
/// let x = 5;
/// ``` 
// pub fn eat_at_restaurant() {
//     println!("Hello from module!");
// }



pub struct NeuralNetwork<const N: usize> {
    pub weights: [f64; N],
    pub bias: f64,
    learning_rate: f64,
}

impl<const N: usize> NeuralNetwork<N> {

    pub fn new() -> NeuralNetwork<N> {

        return NeuralNetwork {
            weights: [0.0; N],
            bias: 0.5,
            learning_rate: 0.5,
        };
    }

    /// Creates a prediction
    ///       
    /// Arguments
    ///
    /// * `input` Input values to create a prediction
    /// 
    /// Return
    /// 
    /// The prediction
    pub fn perdict(&mut self, &input: &[f64; N]) -> f64 {
        let prediction = self.feed_forward(&input);

        return prediction
    }

    /// Trains the neural network
    ///       
    /// Arguments
    ///
    /// * `input` Input values to create a prediction
    /// * `expected` Expected result of the prediction
    /// 
    /// Return
    /// 
    /// Error of the prediction
    pub fn train(&mut self, &input: &[f64; N], expected: &f64) -> f64 {
        let prediction = self.feed_forward(&input);
        let error = self.back_propagation(&input, &prediction, &expected);

        return error
    }


    /// Creates a prediction with the neural network
    /// ``` txt
    ///   Y = S(x1 * w1 + x2 * w2 + ... xN * wN + b)
    ///   x ... input values
    ///   w ... weights
    ///   b ... bias      
    ///   S(x) ... Sigmoid function 
    /// ``` 
    ///                
    /// Arguments
    ///
    /// * `input` Input values to create a prediction
    /// 
    /// Return
    /// 
    /// The prediction
    fn feed_forward(&self, &input: &[f64; N]) -> f64 {
        let bias = 0.5;

        let mut res = bias;

        for i in 0..N {
            res += self.weights[i] * input[i];
        }     

        return sigmoid(res)
    }

    /// Uses a prediction to adjust the weights of the NeuralNetwork
    /// ``` txt
    ///        1   n
    /// MSE = ---  ∑ (Ŷi - Yi)²
    ///        n  i=1
    ///   n ... Number of predicted values
    ///   Ŷ ... Predicted values    
    ///   Y ... Expected values
    /// 
    ///                dError
    /// Wx* = Wx - a (----------) 
    ///                  dWx
    /// 
    /// ``` 
    ///                
    /// Arguments
    ///
    /// * `input` Input values to create a prediction
    /// * `predicted` Predicted result
    /// * `expected` Expected result
    /// 
    /// Return
    /// 
    /// Error value
    fn back_propagation(&mut self,  &input: &[f64; N], &predicted: &f64, &expected: &f64) -> f64 {

        let error = predicted - expected;

        let dcost = error;
        let dpred = sigmoid_derivative(predicted);
        let z_del = dcost * dpred;

        for i in 0..N {
            self.weights[i] = self.weights[i] - self.learning_rate * input[i] * z_del;
        }

        self.bias = self.bias - self.learning_rate * z_del;

        return error
    }
}


/// Activation function
///                
/// Arguments
///
/// * `x`
// fn activation(x: f64) -> f64 {
//     return sigmoid(x)
// }

/// Sigmoid function
/// ``` txt
///            1         
/// S(x) = ----------    
///        1 + e^(-x)   
/// ``` 
///                
/// Arguments
///
/// * `x`
fn sigmoid(x: f64) -> f64 {
    return 1.0 / 
    (1.0 + (-x).exp())
}

/// Derivative of the Sigmoid function
/// ``` txt
/// dS(x)         
/// ----- = S(x) * (1 - S(x))   
///  dx     
/// ``` 
///                
/// Arguments
///
/// * `x`
fn sigmoid_derivative(x: f64) -> f64 {
    return sigmoid(x) * (1.0 - sigmoid(x))
}


