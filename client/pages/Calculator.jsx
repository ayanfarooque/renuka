import { useState } from "react";

export default function Calculator() {
  const [value, setValue] = useState("");

  const handleClick = (v) => {
    if (v === "C") setValue("");
    else if (v === "=") {
      try {
        setValue(eval(value).toString());
      } catch {
        setValue("Error");
      }
    } else {
      setValue(value + v);
    }
  };

  const buttons = [
    "7","8","9","/",
    "4","5","6","*",
    "1","2","3","-",
    "0",".","=","+",
    "C"
  ];

  return (
    <div style={{display:"flex",justifyContent:"center",alignItems:"center",height:"100vh",background:"#f2f2f2"}}>
      <div style={{width:"260px",background:"#fff",padding:"20px",borderRadius:"10px",boxShadow:"0 0 10px rgba(0,0,0,0.2)"}}>
        <input
          value={value}
          readOnly
          style={{width:"100%",height:"50px",fontSize:"20px",marginBottom:"10px",textAlign:"right",padding:"5px"}}
        />
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:"10px"}}>
          {buttons.map((b,i)=>(
            <button
              key={i}
              onClick={()=>handleClick(b)}
              style={{height:"45px",fontSize:"18px",borderRadius:"5px",border:"none",background:"#4CAF50",color:"#fff"}}
            >
              {b}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}