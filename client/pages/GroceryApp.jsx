import { useState } from "react";

const products = [
  { id: 1, name: "Rice", price: 50 },
  { id: 2, name: "Milk", price: 30 },
  { id: 3, name: "Bread", price: 25 },
  { id: 4, name: "Vegetables", price: 40 }
];

export default function GroceryApp() {
  const [cart, setCart] = useState([]);
  const [success, setSuccess] = useState(false);

  const addToCart = (item) => {
    setCart([...cart, item]);
    setSuccess(false);
  };

  const placeOrder = () => {
    setCart([]);
    setSuccess(true);
  };

  const total = cart.reduce((sum, item) => sum + item.price, 0);

  return (
    <div style={{
      minHeight:"100vh",
      display:"flex",
      justifyContent:"center",
      alignItems:"center",
      background:"#f2f2f2",
      fontFamily:"Arial"
    }}>
      <div style={{
        width:"400px",
        background:"#fff",
        padding:"20px",
        borderRadius:"10px",
        boxShadow:"0 0 10px rgba(0,0,0,0.2)"
      }}>
        <h2 style={{textAlign:"center",color:"#2e7d32"}}>
          Grocery Delivery App
        </h2>

        <h3>Available Items</h3>
        {products.map(item => (
          <div key={item.id} style={{display:"flex",justifyContent:"space-between",marginBottom:"10px"}}>
            <span>{item.name} - ₹{item.price}</span>
            <button
              onClick={() => addToCart(item)}
              style={{
                padding:"5px 10px",
                background:"#4CAF50",
                color:"#fff",
                border:"none",
                borderRadius:"5px",
                cursor:"pointer"
              }}
            >
              Add
            </button>
          </div>
        ))}

        <h3>Cart</h3>
        {cart.length === 0 && !success && <p>No items in cart</p>}

        {cart.length > 0 && (
          <ul>
            {cart.map((item, index) => (
              <li key={index}>{item.name} - ₹{item.price}</li>
            ))}
          </ul>
        )}

        <h3>Total: ₹{total}</h3>

        {cart.length > 0 && (
          <button
            onClick={placeOrder}
            style={{
              width:"100%",
              padding:"10px",
              background:"#ff7043",
              color:"#fff",
              border:"none",
              borderRadius:"5px",
              cursor:"pointer"
            }}
          >
            Place Order
          </button>
        )}

        {success && (
          <p style={{
            marginTop:"15px",
            color:"green",
            textAlign:"center",
            fontWeight:"bold"
          }}>
            Order Placed Successfully 
          </p>
        )}
      </div>
    </div>
  );
}