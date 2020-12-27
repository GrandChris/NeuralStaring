#![feature(min_const_generics)]

mod neural_network;
use neural_network::NeuralNetwork;

use rand::Rng;


fn main() {
    println!("Hello, world!");

    let mut network: NeuralNetwork<2> = NeuralNetwork::new();

    let mut data: Vec<(f64, f64, f64)> = Vec::new();

    let mut rng = rand::thread_rng();
    for _i in 0..10 {
        let x1: bool = rng.gen();
        let x2: bool = rng.gen();
        let y = x1 && x2; 
        let vals = (x1 as u32 as f64, x2 as u32 as f64, y as u32 as f64);

        data.push(vals);
    }

    for vals in data {
        let input: [f64; 2] = [vals.0, vals.1];
        let error = network.train(&input, &vals.2);

        println!("{} {} {} {}", error, network.weights[0], network.weights[1], network.bias);
    }

    // neural_network::eat_at_restaurant();

}
