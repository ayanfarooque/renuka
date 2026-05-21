import { useState } from "react";

function useCounter() {
  const [count, setCount] = useState(0);
  const increment = () => setCount(count + 1);
  return [count, increment];
}

export default function HooksDemo() {
  const [count, increment] = useCounter();

  return (
    <div style={{padding:"20px"}}>
      <h2>State Hooks & Custom Hooks</h2>
      <p>Count: {count}</p>
      <button onClick={increment} style={{padding:"8px 15px"}}>
        Increment
      </button>
    </div>
  );
}