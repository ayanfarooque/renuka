import React, { Component } from "react";

function ParentComponent() {
  const message = "Hello from Parent Component";

  return (
    <div style={{padding:"20px",background:"#e8f5e9"}}>
      <h2>Parent Component</h2>
      <ChildComponent message={message} />
    </div>
  );
}

class ChildComponent extends Component {
  render() {
    return (
      <div style={{padding:"15px",marginTop:"10px",background:"#fff3e0"}}>
        <h3>Child Component</h3>
        <p>{this.props.message}</p>
      </div>
    );
  }
}

export default ParentComponent;