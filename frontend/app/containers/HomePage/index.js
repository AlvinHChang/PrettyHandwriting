/*
 * HomePage
 *
 * This is the first thing users see of our App, at the '/' route
 *
 */

import React, {useState} from 'react';
import { FormattedMessage } from 'react-intl';
import messages from './messages';

import VimText from '../../components/VimText/Vim';

export default function HomePage() {
  return (
      <Example/>
  );
}

function Example() {
  // Declare a new state variable, which we'll call "count"
  const [count, setCount] = useState('Inactive');

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount('Active')}>
        Click me
      </button>
    </div>
  );
}
