import React, {useState} from 'react';

class VimText extends React.Component {

    // Declare a new state variable, which we'll call "count"
  render() {
    const [count, setCount] = useState(0);

    return (
          <div>
          <p>You clicked {count} times</p>
          <button onClick={() => setCount(count + 1)}>
            Click me
          </button>
        </div>
    );

  }

}

export default VimText;
