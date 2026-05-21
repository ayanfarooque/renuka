import { useState } from "react";

export default function LoginRegister() {
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm] = useState({ name: "", email: "", password: "" });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    alert(isLogin ? "Login Successful" : "Registration Successful");
  };

  return (
    <div style={{height:"100vh",display:"flex",justifyContent:"center",alignItems:"center",background:"#f0f0f0"}}>
      <div style={{width:"320px",background:"#fff",padding:"20px",borderRadius:"10px",boxShadow:"0 0 10px rgba(0,0,0,0.2)"}}>
        <h2 style={{textAlign:"center"}}>{isLogin ? "Login" : "Register"}</h2>

        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <input
              name="name"
              placeholder="Name"
              value={form.name}
              onChange={handleChange}
              style={{width:"100%",padding:"10px",marginBottom:"10px"}}
            />
          )}

          <input
            name="email"
            placeholder="Email"
            value={form.email}
            onChange={handleChange}
            style={{width:"100%",padding:"10px",marginBottom:"10px"}}
          />

          <input
            type="password"
            name="password"
            placeholder="Password"
            value={form.password}
            onChange={handleChange}
            style={{width:"100%",padding:"10px",marginBottom:"10px"}}
          />

          <button
            type="submit"
            style={{width:"100%",padding:"10px",background:"#4CAF50",color:"#fff",border:"none",borderRadius:"5px"}}
          >
            {isLogin ? "Login" : "Register"}
          </button>
        </form>

        <p style={{textAlign:"center",marginTop:"10px",cursor:"pointer",color:"#007BFF"}}
           onClick={()=>setIsLogin(!isLogin)}>
          {isLogin ? "Create an account" : "Already have an account?"}
        </p>
      </div>
    </div>
  );
}