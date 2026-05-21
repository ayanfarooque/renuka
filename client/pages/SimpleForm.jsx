import { useState } from "react";

export default function SimpleForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    alert(name + " " + email);
  };

  return (
    <div style={{padding:"20px"}}>
      <h2>Simple React Form</h2>

      <form onSubmit={handleSubmit}>
        <input
          placeholder="Name"
          value={name}
          onChange={(e)=>setName(e.target.value)}
          style={{display:"block",marginBottom:"10px",padding:"8px"}}
        />

        <input
          placeholder="Email"
          value={email}
          onChange={(e)=>setEmail(e.target.value)}
          style={{display:"block",marginBottom:"10px",padding:"8px"}}
        />

        <button type="submit" style={{padding:"8px 15px"}}>
          Submit
        </button>
      </form>
    </div>
  );
}