import logo from './logo.svg';
import './App.css';
import React from 'react';
import { Component } from 'react/cjs/react.production.min';
import * as  tf from '@tensorflow/tfjs';
//add(Dense(3, input_dim=5,
const example=tf.tensor([818.8491925,	0	,409.4245962,	0.712143928,	-1]);
class  App extends Component{
  constructor(props) { 
    super(props);
    this.state = {
        posture:""
      }; 
}
  classPosture = ()=>{
    const post ={
      //SD:-1, ED:-1, SED:-1, SS:-1, ES:-1
      SD:418,ED:117,SED:188.2727,SS:0,ES:0
    };
    fetch("http://localhost:4000/posture", {
      method : "post",
      headers : {
        "content-type" : "application/json",
      },
      body : JSON.stringify(post),
    })
    .then((res)=>res.json())
    .then((json)=>{
      this.setState({
        posture : json.text,
      });
    });
  };
  render(){
  return (
    <div>
      <h1>hi</h1>
      <button onClick = {this.classPosture}>Submit</button>
      <h1>{this.state.posture}</h1>
    </div>
  );}
}

export default App;
