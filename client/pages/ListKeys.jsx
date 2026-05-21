import React from "react";

export default function ListKeys() {
  const subjects = ["React", "Node", "MongoDB", "Express"];

  return (
    <div style={{ padding: "20px" }}>
      <h2>Lists and Keys</h2>
      <ul>
        {subjects.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
}