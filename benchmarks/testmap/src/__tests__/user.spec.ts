import { User } from '../user';
it('user', () => { expect(new User('x').name).toBe('x'); });
