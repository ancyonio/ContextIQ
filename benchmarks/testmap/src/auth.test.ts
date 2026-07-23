import { sign } from './auth';
test('sign', () => { expect(sign('a')).toBe('a.sig'); });
