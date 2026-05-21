import { useState } from "react";

export default function EventHandling() {
  const [message, setMessage] = useState("Click a button");

  return (
    <div style={{height:"100vh",display:"flex",justifyContent:"center",alignItems:"center",background:"#f5f5f5"}}>
      <div style={{padding:"20px",background:"#fff",borderRadius:"8px",boxShadow:"0 0 10px rgba(0,0,0,0.2)",textAlign:"center"}}>
        <h2>React Event Handling</h2>
        <p>{message}</p>

        <button
          onClick={() => setMessage("Button Clicked")}
          style={{padding:"10px 15px",margin:"5px"}}
        >
          Click
        </button>

        <button
          onMouseOver={() => setMessage("Mouse Over Button")}
          style={{padding:"10px 15px",margin:"5px"}}
        >
          Hover
        </button>

        <input
          type="text"
          placeholder="Type here"
          onChange={(e) => setMessage(e.target.value)}
          style={{display:"block",marginTop:"10px",padding:"8px",width:"100%"}}
        />
      </div>
    </div>
  );
}