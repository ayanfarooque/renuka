function Header() {
  return <h2 style={{color:"green"}}>Welcome to React</h2>;
}

function Content() {
  return <p>This content is passed using composition</p>;
}

function Container(props) {
  return (
    <div style={{border:"2px solid #000",padding:"15px"}}>
      {props.children}
    </div>
  );
}

export default function Composition() {
  return (
    <div style={{padding:"20px"}}>
      <Container>
        <Header />
        <Content />
      </Container>
    </div>
  );
}