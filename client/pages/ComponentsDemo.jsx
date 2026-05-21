import React, { Component } from "react";

function FunctionComponent() {
  return (
    <div style={{padding:"10px",background:"#e3f2fd",marginBottom:"10px"}}>
      <h2>Function Component</h2>
      <p>This is a functional component</p>
    </div>
  );
}

class ClassComponent extends Component {
  render() {
    return (
      <div style={{padding:"10px",background:"#fce4ec"}}>
        <h2>Class Component</h2>
        <p>This is a class component</p>
      </div>
    );
  }
}

export default function ComponentsDemo() {
  return (
    <div style={{padding:"20px"}}>
      <FunctionComponent />
      <ClassComponent />
    </div>
  );
}