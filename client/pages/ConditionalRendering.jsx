import { useState } from "react";

export default function ConditionalRendering() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  return (
    <div style={{padding:"20px"}}>
      <h2>Conditional Rendering</h2>

      {isLoggedIn ? <p>Welcome User</p> : <p>Please Login</p>}

      <button
        onClick={() => setIsLoggedIn(!isLoggedIn)}
        style={{padding:"8px 15px"}}
      >
        {isLoggedIn ? "Logout" : "Login"}
      </button>
    </div>
  );
}